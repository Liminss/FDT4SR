import torch
import torch.nn as nn
import random
import numpy as np

from modules import Encoder, LayerNorm, Decoder


class FDT4SR(nn.Module):
    def __init__(self, args):
        super(FDT4SR, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        if args.use_position_embeddings:
            self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.item_encoder = Encoder(args)
        self.item_decoder = Decoder(args)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        item_embeddings = self.item_embeddings(sequence)
        #uniform_noise = torch.rand_like(item_embeddings).to(item_embeddings)
        #item_embeddings = item_embeddings * uniform_noise
        #Gaussian_noise = torch.randn_like(item_embeddings).to(item_embeddings)
        #item_embeddings = item_embeddings + Gaussian_noise*0.01
        sequence_emb = item_embeddings
        if self.args.use_position_embeddings:
            seq_length = sequence.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
            position_ids = position_ids.unsqueeze(0).expand_as(sequence)
            position_embeddings = self.position_embeddings(position_ids)
            sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    def extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, sequence_emb, extended_attention_mask):
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        return item_encoded_layers[-1]

    def decode(self, z, extended_attention_mask):
        item_decoder_layers = self.item_decoder(z, extended_attention_mask, output_all_encoded_layers = True)
        return item_decoder_layers[-1]

    def forward(self, input_ids):
        sequence_emb = self.add_position_embedding(input_ids)
        extended_attention_mask = self.extended_attention_mask(input_ids)

        z = self.encode(sequence_emb, extended_attention_mask)
        reconstructed_seq = self.decode(z, extended_attention_mask)
        return reconstructed_seq