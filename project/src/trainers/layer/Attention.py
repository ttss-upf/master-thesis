import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        # key_size = 2 * hidden_size if key_size is None else key_size
        # query_size = hidden_size if query_size is None else query_size
        # self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        # self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        # self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None
        self.q_alphas = None

        # 修改的
        self.query_other = nn.Parameter(torch.FloatTensor(1, 2 * hidden_size),
                                  requires_grad=True)
        self.text_other = nn.Parameter(torch.FloatTensor(1, 2 * hidden_size),
                                        requires_grad=True)
        self.text_attn_layer = nn.Linear(hidden_size + 2 * hidden_size + 1 + 2 * hidden_size, 2 * hidden_size,
                                    bias=True)
        self.query_attn_layer = nn.Linear(hidden_size + 2 * hidden_size + 1, 2 * hidden_size,
                                    bias=True)
        nn.init.uniform_(self.text_other.data, -0.005, 0.005)
        nn.init.uniform_(self.query_other.data, -0.005, 0.005)
        self.ctx2ctx = nn.Linear(2 * hidden_size + 2 * hidden_size, 2 * hidden_size)
        self.query = None

        # ------

    def forward(self, query=None, key=None, value=None, mask=None, q_query=None, q_key=None,
                q_value=None, q_mask=None):
        assert mask is not None, "document mask is required"
        assert q_mask is not None, "query mask is required"

        # query = self.query_layer(query)

        # 原来的
        # scores = self.energy_layer(torch.tanh(query + proj_key))
        # scores = scores.squeeze(2).unsqueeze(1)

        # 修改的
        self.query = query

        # query attention
        query = self.query.repeat(1, q_key.size(1), 1)
        something = torch.cat((q_key, query, self.q_alphas.view(q_key.size(0), q_key.size(1), -1)), dim=-1)
        something = F.tanh(self.query_attn_layer(something))
        scores = torch.zeros(something.size(0), 1, something.size(1), device=device)
        for b in range(something.size(0)):
            thing = something[b, :, :]
            scores[b, :, :] = self.query_other.matmul(thing.T)

        scores.data.masked_fill_(q_mask == 0, -float('inf'))

        q_alphas = F.softmax(scores, dim=-1)
        query_loss = torch.sum(
            torch.min(torch.cat((self.q_alphas.squeeze(1), q_alphas.squeeze(1)), dim=0), -2).values)

        self.q_alphas += q_alphas
        q_context = torch.bmm(q_alphas, q_value)

        # document attention
        query = self.query.repeat(1, key.size(1), 1)
        something = torch.cat(
            (key, query, self.alphas.view(key.size(0), key.size(1), -1), q_context.repeat(1, key.size(1), 1)),
            dim=-1)
        something = F.tanh(self.text_attn_layer(something))
        # self.other = self.other.repeat(1,key.size(1),1)
        # scores = self.other.dot(something)
        scores = torch.zeros(something.size(0), 1, something.size(1), device=device)
        for b in range(something.size(0)):
            thing = something[b, :, :]
            scores[b, :, :] = self.text_other.matmul(thing.T)

        scores.data.masked_fill_(mask == 0, -float('inf'))

        alphas = F.softmax(scores, dim=-1)
        text_loss = torch.sum(
            torch.min(torch.cat((self.alphas.squeeze(1), alphas.squeeze(1)), dim=0), -2).values)
        context = torch.bmm(alphas, value)
        self.alphas += alphas

        # concat document context and query context as the final context
        # context = F.tanh(self.ctx2ctx(torch.cat((context, q_context), dim=-1)))

        return context, alphas, q_alphas, text_loss, query_loss
