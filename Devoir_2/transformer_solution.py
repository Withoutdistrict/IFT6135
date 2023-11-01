import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """
        x = inputs
        return (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sqrt(torch.var(x, dim=-1, unbiased=False, keepdim=True) + self.eps) * self.weight + self.bias

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        self.WQ = nn.Parameter(torch.empty((head_size * num_heads, head_size * num_heads)))
        self.WK = nn.Parameter(torch.empty((head_size * num_heads, head_size * num_heads)))
        self.WV = nn.Parameter(torch.empty((head_size * num_heads, head_size * num_heads)))
        self.WY = nn.Parameter(torch.empty((head_size * num_heads, head_size * num_heads)))

        self.bQ = nn.Parameter(torch.empty(head_size * num_heads))
        self.bK = nn.Parameter(torch.empty(head_size * num_heads))
        self.bV = nn.Parameter(torch.empty(head_size * num_heads))
        self.bY = nn.Parameter(torch.empty(head_size * num_heads))

        self._reset_parameters()

        self.softmax = nn.Softmax(dim=-1)

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.shape == (self.head_size * self.num_heads, self.head_size * self.num_heads):
                nn.init.xavier_uniform_(parameter)
            elif parameter.shape == (self.head_size * self.num_heads,):
                nn.init.zeros_(parameter)

    def get_attention_weights(self, queries, keys, mask=None):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 
           
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """

        attention_weights = queries @ keys.transpose(-1, -2) / math.sqrt(self.head_size)
        if mask is not None:
            s = self.sequence_length - mask.shape[-1]
            if s > 0: mask = F.pad(input=mask, pad=(0, s, 0, 0), mode='constant', value=0)  # ?
            attention_weights.masked_fill_(mask.view(mask.shape[0], 1, 1, mask.shape[1]).expand(-1, -1, mask.shape[1], -1) == 0, float('-inf'))
            # mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)

        attention_weights = self.softmax(attention_weights)

        return attention_weights

    def apply_attention(self, queries, keys, values, mask=None):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. 

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. 
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. 
        """

        return self.merge_heads(self.get_attention_weights(queries, keys, mask) @ values)

    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        B, L = tensor.shape[0], tensor.shape[1]
        return tensor.reshape(shape=[B, L, self.num_heads, -1]).transpose(1, 2)

    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        B, L = tensor.shape[0], tensor.shape[2]
        return tensor.transpose(1, 2).reshape(shape=[B, L, -1])

    def forward(self, hidden_states, mask=None):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.
            
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """

        X = hidden_states

        q = self.split_heads(X @ self.WQ + self.bQ)
        k = self.split_heads(X @ self.WK + self.bK)
        v = self.split_heads(X @ self.WV + self.bV)

        return self.apply_attention(q, k, v, mask) @ self.WY + self.bY


class PostNormAttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, sequence_length, dropout=0.30):
        super(PostNormAttentionBlock, self).__init__()
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
            super(MultiHeadedAttention, self).__init__()
        """

        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim // num_heads, num_heads, sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attention_outputs = self.attn(x, mask)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs + attention_outputs)
        return outputs


class PreNormAttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, sequence_length, dropout=0.0):
        super(PreNormAttentionBlock, self).__init__()

        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim // num_heads, num_heads, sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x_norm_1 = self.layer_norm_1(x)
        attention_outputs = self.attn(x_norm_1, mask)
        outputs = x + attention_outputs

        x_norm_2 = self.layer_norm_2(outputs)
        outputs = self.linear(x_norm_2) + outputs

        return outputs


class Transformer(nn.Module):
    def __init__(self, vocabulary_size=30522, embed_dim=256, hidden_dim=256, num_heads=1, num_layers=2, block='prenorm', dropout=0.3):
        super().__init__()

        # Adding the cls token to the sequnence
        self.sequence_length = 1 + 256
        # Layers/Networks
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        if block == 'prenorm':
            self.transformer = nn.ModuleList([PreNormAttentionBlock(embed_dim, hidden_dim, num_heads, self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        else:
            self.transformer = nn.ModuleList([PostNormAttentionBlock(embed_dim, hidden_dim, num_heads, self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.sequence_length, embed_dim))

    def forward(self, x, mask=None):
        """Transformer

        This is a small version of  Transformer

        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, sequence length)`)
            The input tensor containing text.
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input

        x = self.embedding(x)
        B, T, _ = x.shape

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :T + 1]
        # Add dropout and then the transformer
        # ==========================

        x = self.dropout(x)
        for transformer in self.transformer:
            x = transformer(x, mask)
        # ==========================

        # Take the cls token representation and send it to mlp_head
        # ==========================
        outputs = self.mlp_head(x[:, 0])
        # ==========================
        return outputs
