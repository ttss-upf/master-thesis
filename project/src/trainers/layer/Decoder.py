import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5, bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size  # 512
        self.num_layers = num_layers  # 1
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)

        self.attn2attn = nn.Linear(2 * hidden_size + hidden_size, 2 * hidden_size)
        # self.ff1 = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=True)
        self.attn_gate = nn.Linear(2 * hidden_size + hidden_size, 2 * hidden_size + hidden_size)
        self.info_vector = nn.Linear(2 * hidden_size + hidden_size, 2 * hidden_size + hidden_size)
        self.query_concat_layer = nn.Linear(2 * hidden_size + hidden_size, hidden_size)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(
            hidden_size + 2 * hidden_size + emb_size, hidden_size, bias=False)

    def forward_step(self, prev_embed, text_encoder_hiddens, text_mask, query_encoder_hiddens,
                     query_mask, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D] # decoder的hidden state
        # query_hat = query_encoder_final[-1].unsqueeze(1) # rdf encoder 的最终state, 用来表示rdf的信息
        # query = self.query_concat_layer(torch.cat((query, query_hat), dim=-1))
        context, alphas, q_alphas, text_loss, query_loss = self.attention(query=query,
                                                                          key=text_encoder_hiddens,
                                                                          value=text_encoder_hiddens,
                                                                          mask=text_mask,
                                                                          q_query=query,
                                                                          q_key=query_encoder_hiddens,
                                                                          q_value=query_encoder_hiddens,
                                                                          q_mask=query_mask
                                                                          )

        # attention on attention
        # size (batch_size, 1, 2 * hidden + 2 * hidden)
        # context = torch.cat((context, query), dim=-1)  # 用本身去求算本身
        # ctx1 = F.sigmoid(self.attn_gate(context))
        # ctx2 = self.info_vector(context)
        # context = F.tanh(self.attn2attn(ctx1 * ctx2))
        # attention on attention completed

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output, text_loss, query_loss

    def forward(self, summary_embed, text_encoder_hiddens, text_encoder_final, text_mask,
                summary_mask, query_encoder_hiddens,
                query_encoder_final, query_mask, hidden=None, max_len=None):

        if max_len is None:
            max_len = summary_mask.size(-1)

        if hidden is None:
            hidden = self.init_hidden(text_encoder_final)
        # batch size and seq length. e.g. 4 * 887
        self.attention.alphas = torch.zeros(text_encoder_hiddens.size(0), 1,
                                            text_encoder_hiddens.size(1), device=device)
        self.attention.q_alphas = torch.zeros(query_encoder_hiddens.size(0), 1,
                                              query_encoder_hiddens.size(1), device=device)

        decoder_states = []
        pre_output_vectors = []
        text_total_loss = torch.zeros(1, device=device)
        query_total_loss = torch.zeros(1, device=device)
        # alphas_history = torch.zeros(text_encoder_hiddens.size(0), text_encoder_hiddens.size(1), 1,
        #                              device=device)
        # q_alphas_history = torch.zeros(query_encoder_hiddens.size(0), query_encoder_hiddens.size(1),
        #                                1, device=device)

        for i in range(max_len):
            prev_embed = summary_embed[:, i].unsqueeze(1)
            output, hidden, pre_output, text_loss, query_loss = self.forward_step(prev_embed,
                                                                                  text_encoder_hiddens,
                                                                                  text_mask,
                                                                                  query_encoder_hiddens,
                                                                                  query_mask, hidden
                                                                                  )
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)
            text_total_loss += text_loss
            query_total_loss += query_loss

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors, text_total_loss, query_total_loss

    def init_hidden(self, text_encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if text_encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(text_encoder_final))
