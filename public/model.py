import torch.nn as nn
import torch
import math
import torch.nn.functional as F


# 1. Embedding
class StemLayer(nn.Module):
    def __init__(self, number_input_features, number_output_features, kernel_size, pool_degree):
        super(StemLayer, self).__init__()
        self.stem_conv = nn.Conv1d(in_channels=number_input_features,
                                   out_channels=number_output_features,
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2,
                                   padding_mode="zeros")
        self.stem_norm = nn.BatchNorm1d(num_features=number_output_features)
        self.stem_gelu = nn.GELU()
        self.stem_pool = nn.MaxPool1d(kernel_size=pool_degree)

    def forward(self, x):
        stem_conv = self.stem_conv(x)
        stem_norm = self.stem_norm(stem_conv)
        stem_gelu = self.stem_gelu(stem_norm)
        stem_pool = self.stem_pool(stem_gelu)

        return stem_pool


class StemEmbedding(nn.Module):
    def __init__(self, c_in=4, d_model=288):
        super(StemEmbedding, self).__init__()
        # 4 stem layer
        c_increase = (d_model - c_in) // 4

        self.stem_layer_1 = StemLayer(number_input_features=c_in,
                                      number_output_features=c_in + 1 * c_increase,
                                      kernel_size=5,
                                      pool_degree=2)
        self.stem_layer_2 = StemLayer(number_input_features=c_in + 1 * c_increase,
                                      number_output_features=c_in + 2 * c_increase,
                                      kernel_size=5,
                                      pool_degree=4)
        self.stem_layer_3 = StemLayer(number_input_features=c_in + 2 * c_increase,
                                      number_output_features=c_in + 3 * c_increase,
                                      kernel_size=5,
                                      pool_degree=4)
        self.stem_layer_4 = StemLayer(number_input_features=c_in + 3 * c_increase,
                                      number_output_features=c_in + 4 * c_increase,
                                      kernel_size=5,
                                      pool_degree=4)

    def forward(self, x):
        stem_layer_1 = self.stem_layer_1(x)
        stem_layer_2 = self.stem_layer_2(stem_layer_1)
        stem_layer_3 = self.stem_layer_3(stem_layer_2)
        stem_layer_4 = self.stem_layer_4(stem_layer_3)

        return stem_layer_4.permute(0, 2, 1)


class BasePairEmbedding(nn.Module):
    def __init__(self, c_in=4, d_model=288):
        super(BasePairEmbedding, self).__init__()
        self.base_pair_conv = nn.Conv1d(in_channels=c_in,
                                        out_channels=d_model,
                                        kernel_size=3,
                                        padding=1,
                                        padding_mode="zeros")
        self.base_pair_gelu = nn.GELU()

    def forward(self, x):
        base_pair_conv = self.base_pair_conv(x)
        base_pair_gelu = self.base_pair_gelu(base_pair_conv)
        return base_pair_gelu.permute(0, 2, 1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = torch.zeros(size=(seq_len, d_model),
                                                requires_grad=False).float()

        position = torch.arange(0, seq_len).unsqueeze(dim=1).float()
        div_value = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        self.positional_embedding[:, 0::2] = torch.sin(position * div_value)
        self.positional_embedding[:, 1::2] = torch.cos(position * div_value)

        self.positional_embedding = self.positional_embedding.unsqueeze(dim=0)

    def forward(self):
        positional_embedding = self.positional_embedding
        return positional_embedding


class SampleEmbedding(nn.Module):
    def __init__(self, c_in=4, d_model=288, seq_len=1344, seq_len_up_bound=1344):
        super(SampleEmbedding, self).__init__()
        self.seq_len = seq_len
        self.seq_len_up_bound = seq_len_up_bound

        if seq_len > seq_len_up_bound:
            self.stem_embedding = StemEmbedding(c_in=c_in,
                                                d_model=d_model)
            # pool 2 4 4 4
            seq_len = seq_len // (2 * 4 * 4 * 4)
        else:
            self.base_pair_embedding = BasePairEmbedding(c_in=c_in,
                                                         d_model=d_model)

        self.positional_embedding = PositionalEmbedding(d_model=d_model,
                                                        seq_len=seq_len)

    def forward(self, x):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        positional_embedding = self.positional_embedding()
        positional_embedding = positional_embedding.to(device)
        if self.seq_len > self.seq_len_up_bound:
            stem_embedding = self.stem_embedding(x)
            embedding = stem_embedding + positional_embedding
        else:

            base_pair_embedding = self.base_pair_embedding(x)
            base_pair_embedding = base_pair_embedding.to(device)
            embedding = base_pair_embedding + positional_embedding

        return embedding


# 2. Encoder
class ProbAttention(nn.Module):
    def __init__(self):
        super(ProbAttention, self).__init__()

    def prob_q_k(self, query, key, down_sample_k, n_query):
        b, h, l_key, dim = key.shape
        _, _, l_query, _ = query.shape

        medium_key = key.unsqueeze(-3).expand(b, h, l_query, l_key, dim)

        index_key = torch.randint(high=l_key,
                                  size=(l_query, down_sample_k))

        medium_key = medium_key[:,
                     :,
                     torch.arange(l_query).unsqueeze(1),
                     index_key,
                     :]

        medium_query_key = torch.matmul(query.unsqueeze(-2), medium_key.transpose(-2, -1)).squeeze(-2)

        index_query = medium_query_key.max(-1).values - torch.div(medium_query_key.sum(-1), l_key)
        index_query = index_query.topk(k=n_query,
                                       sorted=False).indices

        query = query[torch.arange(b)[:, None, None],
                torch.arange(h)[None, :, None],
                index_query,
                :]

        query_key = torch.matmul(query, key.transpose(-2, -1))

        return query_key, index_query

    def _ini_value_uniform(self, value, l_query):
        b, h, l_value, dim = value.shape

        value_uniform = value.mean(dim=-2)
        value_uniform = value_uniform.unsqueeze(-2).expand(b, h, l_query, dim).clone()

        return value_uniform

    def prob_similarity_v(self, value_uniform, value, similarity, index):
        b, h, l_value, dim = value_uniform.shape

        a_map = torch.softmax(similarity, dim=-1)

        value_uniform[torch.arange(b)[:, None, None],
        torch.arange(h)[None, :, None],
        index,
        :] = torch.matmul(a_map, value).type_as(value_uniform)

        return value_uniform

    def forward(self, query, key, value):
        b, l_query, h, dim = query.shape
        _, l_key, _, _ = key.shape

        down_sample_k = l_key // 4
        down_sample_q = l_query // 4

        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        value = value.transpose(2, 1)

        similarity, query_index = self.prob_q_k(query=query,
                                                key=key,
                                                down_sample_k=down_sample_k,
                                                n_query=down_sample_q)
        scale = 1. / math.sqrt(dim)
        similarity = similarity * scale

        value_uniform = self._ini_value_uniform(value=value,
                                                l_query=l_query)
        q_k_v = self.prob_similarity_v(value=value,
                                       value_uniform=value_uniform,
                                       similarity=similarity,
                                       index=query_index)
        q_k_v = q_k_v.transpose(2, 1).contiguous()

        return q_k_v


class SelfAttention(nn.Module):
    def __init__(self, d_models=288, n_heads=8):
        super(SelfAttention, self).__init__()
        d_query = d_models // n_heads
        d_key = d_models // n_heads
        d_value = d_models // n_heads

        self.query_linear = nn.Linear(d_models,
                                      d_query * n_heads)
        self.key_linear = nn.Linear(d_models,
                                    d_key * n_heads)
        self.value_linear = nn.Linear(d_models,
                                      d_value * n_heads)

        self.prob_attention = ProbAttention()

        self.heads = n_heads

    def forward(self, query, key, value):
        b, l, _ = query.shape

        query = self.query_linear(query).view(b, l, self.heads, -1)
        key = self.key_linear(key).view(b, l, self.heads, -1)
        value = self.value_linear(value).view(b, l, self.heads, -1)

        prob_attention = self.prob_attention(query, key, value)
        prob_attention = prob_attention.view(b, l, -1)

        return prob_attention


class EncoderLayer(nn.Module):
    def __init__(self, d_models=288, n_heads=8):
        super(EncoderLayer, self).__init__()
        d_ff = d_models * 2

        self.self_attention = SelfAttention(d_models=d_models,
                                            n_heads=n_heads)

        self.conv_ffn_1 = nn.Conv1d(in_channels=d_models,
                                    out_channels=d_ff,
                                    kernel_size=1)
        self.gelu_ffn_1 = nn.GELU()
        self.conv_ffn_2 = nn.Conv1d(in_channels=d_ff,
                                    out_channels=d_models,
                                    kernel_size=1)
        self.gelu_ffn_2 = nn.GELU()

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_models)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_models)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        self_attention = self.self_attention(x, x, x)
        self_attention = x + self.dropout(self_attention)

        layer_norm_1 = self.layer_norm_1(self_attention)
        layer_norm_1 = layer_norm_1.transpose(-1, 1)

        conv_ffn_1 = self.conv_ffn_1(layer_norm_1)
        conv_ffn_1 = self.gelu_ffn_1(conv_ffn_1)

        conv_ffn_2 = self.conv_ffn_2(conv_ffn_1)
        conv_ffn_2 = self.gelu_ffn_2(conv_ffn_2)
        conv_ffn_2 = conv_ffn_2.transpose(-1, 1)

        ffn = conv_ffn_2 + layer_norm_1.transpose(-1, 1)
        layer_norm_2 = self.layer_norm_2(ffn)

        return layer_norm_2


class PoolingLayer(nn.Module):
    def __init__(self, c_in, pool_degree):
        super(PoolingLayer, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=c_in,
                                    out_channels=c_in,
                                    kernel_size=5,
                                    padding=2,
                                    padding_mode="zeros")
        self.norm = nn.BatchNorm1d(num_features=c_in)
        self.elu = nn.ELU()

        self.pooling = nn.MaxPool1d(kernel_size=pool_degree)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv = self.conv_layer(x)
        conv = self.norm(conv)
        conv = self.elu(conv)

        pooling = self.pooling(conv)
        pooling = pooling.permute(0, 2, 1)

        return pooling


class Encoder(nn.Module):
    def __init__(self, d_models=288, n_heads=8):
        super(Encoder, self).__init__()
        self.encoder_layer_1 = EncoderLayer(d_models=d_models,
                                            n_heads=n_heads)
        self.pooling_layer_1 = PoolingLayer(c_in=d_models,
                                            pool_degree=3)

        self.encoder_layer_2 = EncoderLayer(d_models=d_models,
                                            n_heads=n_heads)
        self.pooling_layer_2 = PoolingLayer(c_in=d_models,
                                            pool_degree=2)

        self.encoder_layer_3 = EncoderLayer(d_models=d_models,
                                            n_heads=n_heads)
        self.pooling_layer_3 = PoolingLayer(c_in=d_models,
                                            pool_degree=2)

        self.encoder_layer_4 = EncoderLayer(d_models=d_models,
                                            n_heads=n_heads)
        self.pooling_layer_4 = PoolingLayer(c_in=d_models,
                                            pool_degree=2)

        self.encoder_layer_5 = EncoderLayer(d_models=d_models,
                                            n_heads=n_heads)
        self.pooling_layer_5 = PoolingLayer(c_in=d_models,
                                            pool_degree=2)

        self.encoder_layer_6 = EncoderLayer(d_models=d_models,
                                            n_heads=n_heads)
        self.pooling_layer_6 = PoolingLayer(c_in=d_models,
                                            pool_degree=2)

    def forward(self, x):
        encoder_layer_1 = self.encoder_layer_1(x)
        pooling_layer_1 = self.pooling_layer_1(encoder_layer_1)

        encoder_layer_2 = self.encoder_layer_2(pooling_layer_1)
        pooling_layer_2 = self.pooling_layer_2(encoder_layer_2)

        encoder_layer_3 = self.encoder_layer_3(pooling_layer_2)
        pooling_layer_3 = self.pooling_layer_3(encoder_layer_3)

        encoder_layer_4 = self.encoder_layer_4(pooling_layer_3)
        pooling_layer_4 = self.pooling_layer_4(encoder_layer_4)

        encoder_layer_5 = self.encoder_layer_5(pooling_layer_4)
        pooling_layer_5 = self.pooling_layer_5(encoder_layer_5)

        encoder_layer_6 = self.encoder_layer_6(pooling_layer_5)
        pooling_layer_6 = self.pooling_layer_6(encoder_layer_6)

        return pooling_layer_6


# 3. Bottleneck
class BottleneckLayer(nn.Module):
    def __init__(self, seq_len, d_models=288):
        super(BottleneckLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=d_models,
                              out_channels=d_models // 2,
                              kernel_size=1)
        self.conv_norm = nn.BatchNorm1d(num_features=d_models // 2)
        self.conv_elu = nn.ELU()

        self.pooling = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten(start_dim=1)

        # 720 (131072, 1000), 1008 (1344), 432 (600)
        self.feature_map = {600: 432, 1000: 720, 1344: 1008, 131072: 720}
        self.linear = nn.Linear(in_features=self.feature_map[seq_len],
                                out_features=32)
        self.linear_norm = nn.LayerNorm(normalized_shape=32)
        self.dropout = nn.Dropout(p=0.2)
        self.linear_elu = nn.ELU()

    def forward(self, x):
        conv = self.conv(x)
        conv = self.conv_norm(conv)
        conv = self.conv_elu(conv)

        pooling = self.pooling(conv)

        f = self.flatten(pooling)

        peak_embedding = self.linear(f)
        peak_embedding = self.linear_norm(peak_embedding)
        peak_embedding = self.dropout(peak_embedding)
        peak_embedding = self.linear_elu(peak_embedding)

        return peak_embedding


# 4. Prediction
class Prediction(nn.Module):
    def __init__(self, n_cells):
        super(Prediction, self).__init__()
        self.linear = nn.Linear(in_features=32,
                                out_features=n_cells)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear(x)
        y = self.sigmoid(y)

        return y


# 5. Informer
class Protrait(nn.Module):
    def __init__(self, seq_len=1344, seq_len_up_bound=1344,
                 c_in=4, d_model=288, n_heads=8, n_cells=2034):
        super(Protrait, self).__init__()
        self.seq_len = seq_len
        self.embedding = SampleEmbedding(c_in=c_in,
                                         d_model=d_model,
                                         seq_len=seq_len,
                                         seq_len_up_bound=seq_len_up_bound)
        self.encoder = Encoder(d_models=d_model,
                               n_heads=n_heads)

        self.bottle_neck = BottleneckLayer(d_models=d_model, seq_len=self.seq_len)

        self.prediction = Prediction(n_cells=n_cells)

    def forward(self, x):
        embedding = self.embedding(x)
        encoder = self.encoder(embedding)
        encoder = encoder.permute(0, 2, 1)

        bottle_neck = self.bottle_neck(encoder)

        y = self.prediction(bottle_neck)
        return y
        # return bottle_neck

    @staticmethod
    def cell_embedding(model):
        """
                    :return: cell_embedding.T: shape = [32, n_cells]
                """
        cell_embedding = model.state_dict()['prediction.linear.weight']
        cell_embedding = cell_embedding.cpu().numpy()
        return cell_embedding


# 6. FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_avg=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.size_avg = size_avg

    def forward(self, pred, y):
        """
        :param pred: bach_size * label_len
        :param y: bach_size * label_len
        :return: loss
        """
        log_p = F.logsigmoid(pred)
        p = log_p * y

        loss = -self.alpha * (1 - p) ** self.gamma * log_p

        if self.size_avg:
            return loss.mean()
        else:
            return loss.sum()


