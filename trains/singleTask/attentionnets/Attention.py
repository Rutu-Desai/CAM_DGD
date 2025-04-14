# attention_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F 

class SelfAttentionModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x: (B, T, D)
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)  # (B, T, D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, T, T)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V)  # (B, T, D)
        return self.out_proj(attended)


class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, query_feat, keyval_feat):
        # query_feat: (B, Tq, D)
        # keyval_feat: (B, Tk, D)

        Q = self.q_proj(query_feat)         # (B, Tq, D)
        K = self.k_proj(keyval_feat)        # (B, Tk, D)
        V = self.v_proj(keyval_feat)        # (B, Tk, D)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, Tq, Tk)
        attn_scores = torch.softmax(attn_weights, dim=-1)                 # (B, Tq, Tk)
        attended = torch.matmul(attn_scores, V)                           # (B, Tq, D)

        return self.out_proj(attended)


def apply_attention_modules(text_feat, audio_feat, video_feat, image_feat, embed_dim=128):
    """
    text_feat, audio_feat, video_feat, image_feat: outputs from the shared encoder (B, T, D)
    Returns text-aware contextual feature: (B, T_text, D)
    """
    # Self-attention for each modality
    text_attn = SelfAttentionModule(embed_dim)
    audio_attn = SelfAttentionModule(embed_dim)
    video_attn = SelfAttentionModule(embed_dim)
    image_attn = SelfAttentionModule(embed_dim)

    text_feat = text_attn(text_feat)
    audio_feat = audio_attn(audio_feat)
    video_feat = video_attn(video_feat)
    image_feat = image_attn(image_feat)

    # Stack non-text modalities as context
    context_feat = torch.cat([audio_feat, video_feat, image_feat], dim=1)  # (B, T_context, D)

    # Cross-attention: text attends to other modalities
    cross_attn = CrossAttentionModule(embed_dim)
    text_attended = cross_attn(text_feat, context_feat)  # (B, T_text, D)

    return text_attended
