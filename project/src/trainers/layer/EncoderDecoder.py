import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, text_encoder, query_encoder, decoder, text_embed, query_embed, summary_embed,
                 generator):
        super(EncoderDecoder, self).__init__()
        self.text_encoder = text_encoder
        self.query_encoder = query_encoder
        self.decoder = decoder
        self.text_embed = text_embed
        self.query_embed = query_embed
        self.summary_embed = summary_embed
        self.generator = generator

    def forward(self, text, query, summary, text_mask, query_mask, summary_mask, text_lengths,
                query_lengths, summary_lengths):
        """Take in and process masked text and target sequences."""
        text_encoder_hidden, text_encoder_final = self.text_encode(text, text_mask, text_lengths)
        query_encoder_hidden, query_encoder_final = self.query_encode(query, query_mask,
                                                                      query_lengths)
        return self.decode(text_encoder_hidden, text_encoder_final, text_mask, summary,
                           summary_mask, query_encoder_hidden,
                           query_encoder_final, query_mask)

    def query_encode(self, query, query_mask, query_lengths):
        return self.query_encoder(self.query_embed(query), query_mask, query_lengths)

    def text_encode(self, text, text_mask, text_lengths):
        return self.text_encoder(self.text_embed(text), text_mask, text_lengths)

    def decode(self, text_encoder_hidden, text_encoder_final, text_mask, summary, summary_mask,
               query_encoder_hidden,
               query_encoder_final, query_mask, decoder_hidden=None):
        return self.decoder(
            self.summary_embed(summary),
            text_encoder_hidden,
            text_encoder_final,
            text_mask,
            summary_mask,
            query_encoder_hidden,
            query_encoder_final,
            query_mask,
            hidden=decoder_hidden
        )


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
