import math
import torch


class MHSA(torch.nn.Module):
    """Multi-Head Self-attention layer.

    Args:
        num_heads (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, latent_dim: int, num_heads: int = 8, dropout_rate: float = 0.0):
        super(MHSA, self).__init__()
        assert latent_dim % num_heads == 0

        self.d = latent_dim // num_heads

        self.h = num_heads
        self.linear_q = torch.nn.Linear(latent_dim, latent_dim)

        self.linear_k = torch.nn.Linear(latent_dim, latent_dim)

        self.linear_v = torch.nn.Linear(latent_dim, latent_dim)
        self.linear_out = torch.nn.Linear(latent_dim, latent_dim)
        self.attn_scores = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, num_heads, time1, d).
            torch.Tensor: Transformed key tensor (#batch, num_heads, time2, d).
            torch.Tensor: Transformed value tensor (#batch, num_heads, time2, d).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, self.h, -1, self.d)
        # gimeno's implementation does not use linear_k
        k = key.view(n_batch, self.h, -1, self.d)
        v = self.linear_v(value).view(n_batch, self.h, -1, self.d)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, num_heads, time2, d).
            scores (torch.Tensor): Attention score (#batch, num_heads, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d)
                weighted by the attention score (#batch, time1, time2).

        """
        if mask is not None:
            mask = mask.unsqueeze(1)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn_scores = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn_scores = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn_scores)
        x = torch.matmul(p_attn, value)

        n_batch = value.size(0)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d)

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
            torch.Tensor: Output tensor (#batch, time1, d).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d)
        return self.forward_attention(v, scores, mask)
