"""
Code based on the official implementation of the open-source ESPNet toolkit.
https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/attention.py#L15
"""

import math
import torch


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    Args:
        num_heads (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        num_heads=8,
        dropout_rate=0.0,
        attn_type="cross_embed",
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert query_dim % num_heads == 0
        assert key_dim % num_heads == 0
        assert value_dim % num_heads == 0

        self.attn_type = attn_type

        # -- we assume d_v always equals d_k
        self.d_query = query_dim // num_heads
        self.d_key = key_dim // num_heads
        self.d_value = value_dim // num_heads
        self.d = max(self.d_query, self.d_key)

        self.h = num_heads
        self.linear_q = torch.nn.Linear(query_dim, query_dim)

        if "cross" in self.attn_type:
            self.linear_k = None
        else:
            self.linear_k = torch.nn.Linear(key_dim, key_dim)

        self.linear_v = torch.nn.Linear(value_dim, value_dim)
        self.linear_out = torch.nn.Linear(key_dim, key_dim)
        self.attn_scores = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, num_heads, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, num_heads, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, num_heads, time2, d_k).

        """
        n_batch = query.size(0)

        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_query)
        if self.linear_k is not None:
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_key)
        k = key.view(n_batch, -1, self.h, self.d_key)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_value)

        q = q.transpose(1, 2)  # (batch, head, time1, d_query)
        k = k.transpose(1, 2)  # (batch, head, time2, d_key)
        v = v.transpose(1, 2)  # (batch, head, time2, d_value)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, num_heads, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, num_heads, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = torch.finfo(scores.dtype).min
            if self.attn_type in ["cross_time"]:
                mask = mask.permute(0, 1, 3, 2)
            scores = scores.masked_fill(mask, min_value)
            self.attn_scores = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            # -- cross ssl embed: (batch, 1024, 619)
            self.attn_scores = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn_scores)

        # -- post-score dimension adjustments
        if self.attn_type in ["cross_embed", "cross_time"]:
            p_attn = p_attn.transpose(-2, -1)

        x = torch.matmul(p_attn, value)

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_key)

        return self.linear_out(x)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)

        # -- pre-score dimension adjustments
        if self.attn_type == "cross_embed":
            q = q.transpose(-2, -1)
            v = v.transpose(-2, -1)
        elif self.attn_type != "cross_time":
            k = k.transpose(-2, -1)

        scores = torch.matmul(q, k) / math.sqrt(self.d)

        return self.forward_attention(v, scores, mask)
