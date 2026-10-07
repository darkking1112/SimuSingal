"""``cv_trn`` 实现：``custom/CV_TRN.py`` 的维护副本（原文件不改）。

相对原文件的差异**只有三处**，模型结构与计算不变：

1. ``from timm.layers import trunc_normal_`` → ``from ._nn_utils import trunc_normal_``
   （等价实现，避免为一个初始化函数引入 timm 依赖）；
2. 构造期不再读 CUDA 可用性写死设备（``cls_token``、RPE 表与索引、注意力 ``scale``
   缓冲区都建在 CPU 上），设备一律交由统一的 ``.to(device)`` 控制；
3. 文件末尾追加 ``build()`` 工厂（目录参数 → 模型；输入 ``(B, 2, N)`` 由 :func:`adapt`
   转成原实现要求的 ``(B, N, 2)``），并去掉原文件的 ``__main__`` 演示块。

结构与参数约束的唯一声明在 ``signal_analysis.algorithms.amc.ai_model.cv_trn``。
"""

# --------------------------------------------------------
# CV-TRN: Complex-Valued Transformer for Automatic Modulation Recognition
# Implementation based on the paper:
# "A Complex-Valued Transformer for Automatic Modulation Recognition"
# IEEE Internet of Things Journal, Vol. 11, No. 12, June 2024
#
# Key features:
# 1. Individual I/Q supervision with shared parameters (ISCM)
# 2. Complex-Valued Multi-Head Self-Attention (CMHSA)
# 3. Relative Position Embedding (RPE)
# 4. Random Phase Offset (RPO) data augmentation
# 5. DB-GLU Feed-Forward Network
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from ._nn_utils import trunc_normal_
import math


class RelativePositionBias(nn.Module):
    """
    Relative Position Embedding for 1D sequences
    Based on Swin Transformer's relative position bias
    """
    def __init__(self, num_heads, seq_length):
        super(RelativePositionBias, self).__init__()
        self.num_heads = num_heads
        self.seq_length = seq_length
        
        # Relative position bias table: (2*N-1) relative positions for each head
        # 在 CPU 上创建，由统一的 .to(device) 搬到设备（不在构造期读 CUDA 可用性）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_heads, 2 * seq_length - 1)
        )
        
        # Create relative position index
        coords = torch.arange(seq_length)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N)
        relative_coords = relative_coords + seq_length - 1  # Shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self):
        # (N, N) -> (h, N, N)
        relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index.view(-1)].view(
            self.num_heads, self.seq_length, self.seq_length
        )
        return relative_position_bias


class CMHSA_block(nn.Module):
    """
    Complex-Valued Multi-Head Self-Attention Block
    
    Key idea: I and Q components interact through complex correlation
    - Attn_real = (Q_i * K_i^T + Q_q * K_q^T) / sqrt(d_p)
    - Attn_imag = (Q_q * K_i^T - Q_i * K_q^T) / sqrt(d_p)
    - Output uses complex multiplication
    """
    def __init__(self,
                 d_model=64,          # Token embedding dimension (dt)
                 d_fix_qk=16,         # Embedding dimension for Q/K (dp)
                 d_fix_v=16,          # Embedding dimension for V (dp)
                 n_head=4,            # Number of heads (h)
                 seq_length=33,       # Sequence length (F+1 with class token)
                 dropout=0.,
                 bias=False,
                 talking=True,        # Talking-head attention
                 use_rpe=True):       # Use relative position embedding
        super(CMHSA_block, self).__init__()
        self.dm = d_model
        self.df_qk = d_fix_qk
        self.df_v = d_fix_v
        self.h = n_head
        self.use_rpe = use_rpe
        
        # 在 CPU 上创建，由统一的 .to(device) 搬到设备（不在构造期读 CUDA 可用性）
        self.register_buffer('scale', torch.tensor(d_fix_qk, dtype=torch.float32) ** 0.5)
        
        # Linear projections for Q, K, V (shared for I and Q components)
        self.to_q = nn.Linear(d_model, d_fix_qk * n_head, bias=bias)
        self.to_k = nn.Linear(d_model, d_fix_qk * n_head, bias=bias)
        self.to_v = nn.Linear(d_model, d_fix_v * n_head, bias=bias)
        
        # Talking-heads projection
        self.proj_bf = nn.Conv2d(n_head, n_head, (1, 1), bias=False) if talking else nn.Identity()
        
        # Output projection
        self.proj_v = nn.Linear(d_fix_v * n_head, d_model, bias=bias)
        
        # Relative position embedding
        if use_rpe:
            self.rpe = RelativePositionBias(n_head, seq_length)
        
        self.softmax = nn.Softmax(dim=-1)
        self.Dp_attn = nn.Dropout(dropout)
        self.Dp_v = nn.Dropout(dropout)

    def forward(self, x_i, x_q):
        """
        Args:
            x_i: In-phase component (B, P, D)
            x_q: Quadrature component (B, P, D)
        Returns:
            x_out_i, x_out_q: Output I/Q components
        """
        B, P, D = x_i.shape
        
        # Generate Q, K, V for both I and Q components (shared projection weights)
        # Shape: (B, P, h, df) -> (B, h, P, df)
        Q_i = self.to_q(x_i).reshape(B, P, self.h, self.df_qk).permute(0, 2, 1, 3)
        Q_q = self.to_q(x_q).reshape(B, P, self.h, self.df_qk).permute(0, 2, 1, 3)
        
        K_i = self.to_k(x_i).reshape(B, P, self.h, self.df_qk).permute(0, 2, 1, 3)
        K_q = self.to_k(x_q).reshape(B, P, self.h, self.df_qk).permute(0, 2, 1, 3)
        
        V_i = self.to_v(x_i).reshape(B, P, self.h, self.df_v).permute(0, 2, 1, 3)
        V_q = self.to_v(x_q).reshape(B, P, self.h, self.df_v).permute(0, 2, 1, 3)
        
        # Complex correlation for attention (Eq. 6 in paper)
        # Attn_real = (Q_i * K_i^T + Q_q * K_q^T) / sqrt(d_p)
        # Attn_imag = (Q_q * K_i^T - Q_i * K_q^T) / sqrt(d_p)
        Attn_real = (Q_i @ K_i.transpose(-2, -1) + Q_q @ K_q.transpose(-2, -1)) / self.scale
        Attn_imag = (Q_q @ K_i.transpose(-2, -1) - Q_i @ K_q.transpose(-2, -1)) / self.scale
        
        # Add relative position embedding
        if self.use_rpe:
            rpe_bias = self.rpe()  # (h, P, P)
            Attn_real = Attn_real + rpe_bias.unsqueeze(0)
            Attn_imag = Attn_imag + rpe_bias.unsqueeze(0)
        
        # Talking-heads (information exchange between heads)
        Attn_real = self.proj_bf(Attn_real)
        Attn_imag = self.proj_bf(Attn_imag)
        
        # Softmax normalization
        Attn_real = self.softmax(Attn_real)
        Attn_imag = self.softmax(Attn_imag)
        
        # Dropout
        Attn_real = self.Dp_attn(Attn_real)
        Attn_imag = self.Dp_attn(Attn_imag)
        
        # Complex multiplication for output (Eq. 9 in paper)
        # X_out_i = Attn_real * V_i - Attn_imag * V_q
        # X_out_q = Attn_real * V_q + Attn_imag * V_i
        X_out_i = (Attn_real @ V_i - Attn_imag @ V_q).transpose(1, 2).reshape(B, P, self.df_v * self.h)
        X_out_q = (Attn_real @ V_q + Attn_imag @ V_i).transpose(1, 2).reshape(B, P, self.df_v * self.h)
        
        # Output projection
        x_out_i = self.Dp_v(self.proj_v(X_out_i))
        x_out_q = self.Dp_v(self.proj_v(X_out_q))
        
        return x_out_i, x_out_q


class DB_GLU_block(nn.Module):
    """
    Dual-Branch Gated Linear Unit for FFN
    Same as FEA-T implementation
    """
    def __init__(self,
                 d_model=64,
                 dim_feedforward=128,
                 dropout=0.,
                 activate='gelu'):
        super(DB_GLU_block, self).__init__()
        self.dim_f = dim_feedforward // 2
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward // 2, d_model)
        
        if activate == 'relu':
            self.Act = nn.ReLU()
        elif activate == 'sigmoid':
            self.Act = nn.Sigmoid()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        x_ = self.linear1(x)
        x_1, x_2 = x_[..., :self.dim_f], x_[..., self.dim_f:]
        # DB-GLU: both branches act as gate and information
        return self.linear2(self.dropout(x_1 * self.Act(x_2) + x_2 * self.Act(x_1)))


class CV_Transformer_layer(nn.Module):
    """
    Complex-Valued Transformer Encoder Layer
    """
    def __init__(self,
                 d_model=64,
                 d_fix_qk=64,
                 d_fix_v=64,
                 d_mid=128,
                 n_head=4,
                 seq_length=33,
                 dropout=0.,
                 bias=False,
                 talking=True,
                 use_rpe=True,
                 activate='gelu'):
        super(CV_Transformer_layer, self).__init__()
        
        self.cmhsa = CMHSA_block(
            d_model=d_model,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            n_head=n_head,
            seq_length=seq_length,
            dropout=dropout,
            bias=bias,
            talking=talking,
            use_rpe=use_rpe
        )
        
        # FFN (shared for I and Q)
        self.ffn = DB_GLU_block(
            d_model=d_model,
            dim_feedforward=d_mid,
            dropout=dropout,
            activate=activate
        )
        
        # Layer normalization (shared for I and Q)
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x_i, x_q):
        # CMHSA with residual connection
        x_sa_i, x_sa_q = self.cmhsa(x_i, x_q)
        x_i = self.norm_1(x_i + x_sa_i)
        x_q = self.norm_1(x_q + x_sa_q)
        
        # FFN with residual connection (shared FFN for I and Q)
        x_i = self.norm_2(x_i + self.ffn(x_i))
        x_q = self.norm_2(x_q + self.ffn(x_q))
        
        return x_i, x_q


class FrameWiseEmbedding(nn.Module):
    """
    Frame-wise Embedding Module using 1D CNN
    Divides signal into overlapping frames
    """
    def __init__(self,
                 frame_length=32,
                 step_size=16,
                 d_model=64):
        super(FrameWiseEmbedding, self).__init__()
        self.frame_length = frame_length
        self.step_size = step_size
        
        # 1D convolution for frame-wise embedding
        self.embedding = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=frame_length,
            stride=step_size,
            bias=False
        )

    def forward(self, x):
        """
        Args:
            x: Input signal (B, N) - single channel (I or Q)
        Returns:
            tokens: (B, F, d_model) where F is number of frames
        """
        B, N = x.shape
        x = x.unsqueeze(1)  # (B, 1, N)
        tokens = self.embedding(x)  # (B, d_model, F)
        tokens = tokens.transpose(1, 2)  # (B, F, d_model)
        return tokens


class CV_TRN(nn.Module):
    """
    Complex-Valued Transformer Network for Automatic Modulation Recognition
    
    Key features:
    1. Individual I/Q input with shared parameters
    2. Complex-Valued MHSA (CMHSA)
    3. Relative Position Embedding (RPE)
    4. Random Phase Offset (RPO) data augmentation
    5. DB-GLU FFN
    """
    def __init__(self,
                 frame_length=32,        # Frame length L
                 step_size=16,           # Step size R (typically L/2 or oversampling ratio)
                 d_model=64,             # Token embedding dimension dt
                 d_fix_qk=64,            # Q/K embedding dimension dp
                 d_fix_v=64,             # V embedding dimension (same as dp)
                 d_mid=128,              # FFN hidden dimension df
                 seq_length=1024,        # Input signal length N
                 n_head=4,               # Number of attention heads h
                 dropout=0.,
                 layer_num=4,            # Number of transformer layers M
                 num_class=24,           # Number of modulation classes
                 bias=False,
                 talking=True,           # Talking-head attention
                 use_rpe=True,           # Relative position embedding
                 use_rpo=True,           # Random phase offset augmentation
                 activation='gelu'):
        super(CV_TRN, self).__init__()
        
        self.use_rpo = use_rpo
        
        # Calculate number of frames
        n_frames = (seq_length - frame_length) // step_size + 1
        total_seq_length = n_frames + 1  # +1 for class token
        
        # Frame-wise embedding (shared for I and Q)
        self.embedding = FrameWiseEmbedding(
            frame_length=frame_length,
            step_size=step_size,
            d_model=d_model
        )
        
        # Class token (shared for I and Q)：在 CPU 上创建，由统一的 .to(device) 搬到设备
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            CV_Transformer_layer(
                d_model=d_model,
                d_fix_qk=d_fix_qk,
                d_fix_v=d_fix_v,
                d_mid=d_mid,
                n_head=n_head,
                seq_length=total_seq_length,
                dropout=dropout,
                bias=bias,
                talking=talking,
                use_rpe=use_rpe,
                activate=activation
            )
            for _ in range(layer_num)
        ])
        
        # Classification head
        # Concatenate class tokens from I and Q branches
        self.classifier = nn.Linear(d_model * 2, num_class)
        
        # Initialize weights
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_phase_offset(self, x_i, x_q):
        """
        Apply random phase offset for data augmentation (Eq. 17 in paper)
        s_offset = s * e^(2*pi*r*i)
        """
        if not self.training or not self.use_rpo:
            return x_i, x_q
        
        B = x_i.shape[0]
        # Generate random phase offset ratio for each sample in batch
        r = torch.rand(B, 1, device=x_i.device) * 1.0  # r in [0, 1]
        phase = 2 * math.pi * r
        
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        
        # Apply phase rotation
        x_i_new = x_i * cos_phase - x_q * sin_phase
        x_q_new = x_i * sin_phase + x_q * cos_phase
        
        return x_i_new, x_q_new

    def forward(self, x, state=None):
        """
        Args:
            x: Input signal (B, 1, N, 2) or (B, N, 2)
               Last dimension: [I, Q] components
        Returns:
            logits: Classification logits (B, num_class)
        """
        # Handle different input shapes
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, N, 2) -> (B, N, 2)
        
        B = x.shape[0]
        
        # Separate I and Q components
        x_i = x[..., 0]  # (B, N)
        x_q = x[..., 1]  # (B, N)
        
        # Apply random phase offset augmentation
        x_i, x_q = self.random_phase_offset(x_i, x_q)
        
        # Frame-wise embedding (shared embedding layer)
        tokens_i = self.embedding(x_i)  # (B, F, d_model)
        tokens_q = self.embedding(x_q)  # (B, F, d_model)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens_i = torch.cat([cls_token, tokens_i], dim=1)  # (B, F+1, d_model)
        tokens_q = torch.cat([cls_token, tokens_q], dim=1)  # (B, F+1, d_model)
        
        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            tokens_i, tokens_q = layer(tokens_i, tokens_q)
        
        # Extract class tokens and concatenate
        cls_i = tokens_i[:, 0]  # (B, d_model)
        cls_q = tokens_q[:, 0]  # (B, d_model)
        cls_feature = torch.cat([cls_i, cls_q], dim=-1)  # (B, d_model*2)
        
        # Classification
        logits = self.classifier(cls_feature)
        
        return logits


def get_config_rml2016():
    """
    Configuration for RML2016.10a dataset
    Signal length: 128, Oversampling ratio: 8
    """
    config = {
        'frame_length': 16,       # L = 2 * R
        'step_size': 8,           # R = oversampling ratio
        'd_model': 64,            # dt
        'd_fix_qk': 64,           # dp
        'd_fix_v': 64,            # dp
        'd_mid': 128,             # df
        'seq_length': 128,        # N
        'n_head': 4,              # h
        'dropout': 0.0,
        'layer_num': 4,           # M  default: 4
        'num_class': 5,          # Number of modulation types
        'bias': False,
        'talking': True,
        'use_rpe': True,
        'use_rpo': True,
        'activation': 'gelu'
    }
    return config


def get_config_rml2018():
    """
    Configuration for RML2018.01a dataset
    Signal length: 1024, Oversampling ratio: 16
    """
    config = {
        'frame_length': 32,       # L = 2 * R
        'step_size': 16,          # R = oversampling ratio
        'd_model': 64,            # dt
        'd_fix_qk': 64,           # dp
        'd_fix_v': 64,            # dp
        'd_mid': 128,             # df
        'seq_length': 1024,       # N
        'n_head': 4,              # h
        'dropout': 0.0,
        'layer_num': 4,           # M
        'num_class': 24,          # Number of modulation types
        'bias': False,
        'talking': True,
        'use_rpe': True,
        'use_rpo': True,
        'activation': 'gelu'
    }
    return config


def numParams(net):
    """Count number of trainable parameters"""
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


# ---------------------------------------------------------------- 项目接入层


def build(*, num_classes: int, samples, params: dict) -> nn.Module:
    """按目录参数构建模型（输入 ``(B, 2, N)``、输出 logits）。

    ``seq_length`` 必须等于数据集窗口长度（帧数、位置编码表与该长度绑定），
    因此由 ``samples`` 传入而不是独立可调参数。
    """
    if samples is None:
        raise ValueError("模型 cv_trn 的结构依赖窗口长度：请提供 samples"
                         "（训练链路会从数据集契约传入）")
    model = CV_TRN(frame_length=int(params["frame_length"]),
                   step_size=int(params["step_size"]),
                   d_model=int(params["d_model"]),
                   d_fix_qk=int(params["d_model"]),
                   d_fix_v=int(params["d_model"]),
                   d_mid=int(params["d_mid"]),
                   seq_length=int(samples),
                   n_head=int(params["n_head"]),
                   dropout=float(params["dropout"]),
                   layer_num=int(params["layer_num"]),
                   num_class=int(num_classes),
                   use_rpe=bool(params["use_rpe"]),
                   use_rpo=bool(params["use_rpo"]))
    from ._adapters import CHANNELS_LAST, adapt

    return adapt(model, CHANNELS_LAST)
