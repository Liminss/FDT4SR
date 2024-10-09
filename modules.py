import os
import copy
import math
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

"activation function"
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


def Absolute_positional_embedding(sentence_length, dim):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec = encoded_vec.reshape([sentence_length, dim])
    for pos in range(sentence_length):
        encoded_vec[pos][::2] = np.sin(encoded_vec[pos][::2])
        encoded_vec[pos][1::2] = np.cos(encoded_vec[pos][1::2])
    return torch.tensor(encoded_vec, dtype=torch.float32)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Encoder_SelfAttention(nn.Module):
    def __init__(self, args):
        super(Encoder_SelfAttention, self).__init__()
        self.args =args
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        if args.use_order:
            self.order_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.activation = nn.Sigmoid()
        if args.use_distance:
            self.distance_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.scalar = nn.Parameter(torch.randn(1))

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.filter = FilterLayer(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        batch_size = input_tensor.shape[0]
        max_seq_len = input_tensor.shape[1]

        q_vec = query_layer.unsqueeze(3).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        k_vec = key_layer.unsqueeze(2).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        q_k = torch.cat((q_vec, k_vec), dim=-1)

        error_order = torch.zeros(attention_scores.shape).to(attention_scores.device)
        error_distance = torch.zeros(attention_scores.shape).to(attention_scores.device)
        if self.args.use_order:
            gd_order = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).unsqueeze(0).unsqueeze(0)
            gd_order = gd_order.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(input_tensor.device)
            pr_order = self.activation(self.order_affine(q_k).squeeze(-1))
            error_order = torch.log(pr_order + 1e-24) * gd_order + torch.log(1 - pr_order + 1e-24) * (1 - gd_order)
        if self.args.use_distance:
            row_index = torch.arange(0, max_seq_len, 1).unsqueeze(0).expand((max_seq_len, max_seq_len))
            col_index = torch.arange(0, max_seq_len, 1).unsqueeze(1).expand((max_seq_len, max_seq_len))
            gd_distance = torch.log(torch.abs(row_index - col_index) + 1).unsqueeze(0).unsqueeze(0)
            gd_distance = gd_distance.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(input_tensor.device)
            pr_distance = self.distance_affine(q_k).squeeze(-1)
            error_distance = -torch.square(gd_distance - pr_distance) * torch.square(self.scalar) / 2
        attention_scores = attention_scores + error_order + error_distance

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.filter(context_layer)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Decoder_SelfAttention(nn.Module):
    def __init__(self, args):
        super(Decoder_SelfAttention, self).__init__()
        self.args = args
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        if args.use_order:
            self.order_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.activation = nn.Sigmoid()
        if args.use_distance:
            self.distance_affine = nn.Linear(2 * self.attention_head_size, 1)
            self.scalar = nn.Parameter(torch.randn(1))

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        batch_size = input_tensor.shape[0]
        max_seq_len = input_tensor.shape[1]

        q_vec = query_layer.unsqueeze(3).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        k_vec = key_layer.unsqueeze(2).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        q_k = torch.cat((q_vec, k_vec), dim=-1)

        error_order = torch.zeros(attention_scores.shape).to(attention_scores.device)
        error_distance = torch.zeros(attention_scores.shape).to(attention_scores.device)
        if self.args.use_order:
            gd_order = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).unsqueeze(0).unsqueeze(0)
            gd_order = gd_order.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_order = self.activation(self.order_affine(q_k).squeeze(-1))
            error_order = torch.log(pr_order + 1e-24) * gd_order + torch.log(1 - pr_order + 1e-24) * (1 - gd_order)
        if self.args.use_distance:
            row_index = torch.arange(0, max_seq_len, 1).unsqueeze(0).expand((max_seq_len, max_seq_len))
            col_index = torch.arange(0, max_seq_len, 1).unsqueeze(1).expand((max_seq_len, max_seq_len))
            gd_distance = torch.log(torch.abs(row_index - col_index) + 1).unsqueeze(0).unsqueeze(0)
            gd_distance = gd_distance.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_distance = self.distance_affine(q_k).squeeze(-1)
            error_distance = -torch.square(gd_distance - pr_distance) * torch.square(self.scalar) / 2
        attention_scores = attention_scores + error_order + error_distance

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length, args.hidden_size//2 + 1, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=-1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=hidden, dim=-1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.inner_size)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act
        self.dense_2 = nn.Linear(args.inner_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(input_tensor + hidden_states)
        return hidden_states


class Encoder_Layer(nn.Module):
    def __init__(self, args):
        super(Encoder_Layer, self).__init__()
        self.attention = Encoder_SelfAttention(args)
    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        return attention_output

class Decoder_Layer(nn.Module):
    def __init__(self, args):
        super(Decoder_Layer, self).__init__()
        self.attention = Decoder_SelfAttention(args)
        self.ffn = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(attention_output)
        return ffn_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Encoder_Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        layer = Decoder_Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_decoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_decoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_decoder_layers.append(hidden_states)
        return all_decoder_layers


class ItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF_IUF', dataset_name=None):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_dict used for itemCF and itemCF-IUF
        self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path='./similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        train_data = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-2]
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        print("Step 1: Compute Statistics")
        data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
        for idx, (u, items) in data_iter:
            if self.model_name == 'ItemCF':
                for i in items.keys():
                    N.setdefault(i, 0)
                    N[i] += 1
                    for j in items.keys():
                        if i == j:
                            continue
                        C.setdefault(i, {})
                        C[i].setdefault(j, 0)
                        C[i][j] += 1
            elif self.model_name == 'ItemCF_IUF':
                for i in items.keys():
                    N.setdefault(i, 0)
                    N[i] += 1
                    for j in items.keys():
                        if i == j:
                            continue
                        C.setdefault(i, {})
                        C[i].setdefault(j, 0)
                        C[i][j] += 1 / math.log(1 + len(items) * 1.0)
        self.itemSimBest = dict()
        print("Step 2: Compute co-rate matrix")
        c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
        for idx, (cur_item, related_items) in c_iter:
            self.itemSimBest.setdefault(cur_item, {})
            for related_item, score in related_items.items():
                self.itemSimBest[cur_item].setdefault(related_item, 0)
                self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
        self._save_dict(self.itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        with open(similarity_model_path, 'rb') as read_file:
            similarity_dict = pickle.load(read_file)
        return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        """TODO: handle case that item not in keys"""
        if str(item) in self.similarity_model:
            top_k_items_with_score = sorted(self.similarity_model[str(item)].items(), key=lambda x: x[1], reverse=True)[0:top_k]
            if with_score:
                return list(
                    map(lambda x: (int(x[0]), (float(x[1]) - self.min_score) / (self.max_score - self.min_score)), top_k_items_with_score))
            return list(map(lambda x: int(x[0]), top_k_items_with_score))
        elif int(item) in self.similarity_model:
            top_k_items_with_score = sorted(self.similarity_model[int(item)].items(), key=lambda x: x[1], reverse=True)[0:top_k]
            if with_score:
                return list(
                    map(lambda x: (int(x[0]), (float(x[1]) - self.min_score) / (self.max_score - self.min_score)), top_k_items_with_score))
            return list(map(lambda x: int(x[0]), top_k_items_with_score))
        else:
            item_list = list(self.similarity_model.keys())
            random_items = random.sample(item_list, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))