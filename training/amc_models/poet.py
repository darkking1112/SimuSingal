"""``poet`` 实现：``custom/POET.py`` 的维护副本（原文件不改）。

相对原文件的差异**只有两处**，模型结构与计算不变：

1. ``from timm.models.layers import trunc_normal_, DropPath`` →
   ``from ._nn_utils import trunc_normal_, DropPath``（等价实现，避免 timm 依赖）；
2. 文件末尾追加 ``build()`` 工厂（目录参数 → 模型；输入 ``(B, 2, N)`` 由 :func:`adapt`
   转成原实现要求的 ``(B, N, 2)``），并去掉原文件的 ``__main__`` 演示块。

训练期增强（``PhysicsAwareAugmentation``）只在 ``self.training`` 时生效，
``InstanceAGC`` 是确定性计算（逐样本功率归一化），二者都保留。
结构与参数约束的唯一声明在 ``signal_analysis.algorithms.amc.ai_model.poet``。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._nn_utils import trunc_normal_, DropPath


# =============================================================
# 1. Instance AGC
# =============================================================
class InstanceAGC(nn.Module):
    """Instance-level automatic gain control with robust cross-scale normalization."""
    def __init__(self, floor_eps=1e-12, rel_eps=1e-4, learnable_gain=True):
        super().__init__()
        self.floor_eps = floor_eps
        self.rel_eps = rel_eps
        if learnable_gain:
            self.gain_net = nn.Sequential(
                nn.Linear(2, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            nn.init.zeros_(self.gain_net[-1].weight)
            nn.init.zeros_(self.gain_net[-1].bias)
        else:
            self.gain_net = None
        self._last_log_power = None

    def forward(self, x_i, x_q):
        inst_power = x_i.pow(2) + x_q.pow(2)
        mean_power = inst_power.mean(dim=-1, keepdim=True)

        eps = torch.clamp(mean_power * self.rel_eps, min=self.floor_eps)
        scale = torch.rsqrt(mean_power + eps)
        x_i_out = x_i * scale
        x_q_out = x_q * scale

        if self.gain_net is not None:
            log_power = torch.log(mean_power + self.floor_eps)
            peak_power = inst_power.max(dim=-1, keepdim=True)[0]
            papr = peak_power / (mean_power + eps)
            log_papr = torch.log(papr + 1e-6)

            log_power_centered = log_power - log_power.mean()
            stats = torch.cat([log_power_centered, log_papr], dim=-1)
            gain = 1.0 + self.gain_net(stats)
            x_i_out = x_i_out * gain
            x_q_out = x_q_out * gain

        self._last_log_power = torch.log(mean_power.detach() + self.floor_eps)
        return x_i_out, x_q_out


# =============================================================
# 2. CrossScaleAttention
# =============================================================
class CrossScaleAttention(nn.Module):
    def __init__(self, channels, num_scales, hidden=None, init_temperature=1.5):
        super().__init__()
        self.num_scales = num_scales
        h = hidden or max(channels // 2, 8)
        self.score = nn.Sequential(
            nn.Conv1d(channels, h, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(h, 1, kernel_size=1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self._last_attn = None

    def forward(self, stacked):
        B, C, S, L = stacked.shape
        x = stacked.permute(0, 2, 1, 3).reshape(B * S, C, L)
        scores = self.score(x).reshape(B, S, L)
        temperature = self.log_temperature.exp().clamp(min=0.25, max=4.0)
        attn = F.softmax(scores / temperature, dim=1)
        self._last_attn = attn
        return (stacked * attn.unsqueeze(1)).sum(dim=2)


# =============================================================
# 3. Scale-equivariant embedding
# =============================================================
class ScaleEquivariantConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5,
                 scales=(1, 2, 4, 8), stride=1):
        super().__init__()
        self.scales = tuple(scales)
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        num_groups = min(8, out_ch)
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.bn = nn.ModuleList([nn.GroupNorm(num_groups, out_ch) for _ in self.scales])
        self.fusion = CrossScaleAttention(out_ch, len(self.scales))

    def forward(self, x):
        target_len = None
        feats = []
        for idx, s in enumerate(self.scales):
            pad = (self.kernel_size - 1) * s // 2
            y = F.conv1d(x, self.weight, bias=None,
                         stride=self.stride, padding=pad, dilation=s)
            y = self.bn[idx](y)
            feats.append(y)
            target_len = y.shape[-1] if target_len is None else min(target_len, y.shape[-1])
        feats = [F.adaptive_avg_pool1d(f, target_len) for f in feats]
        stacked = torch.stack(feats, dim=2)
        return self.fusion(stacked)


class ScaleEquivariantEmbedding(nn.Module):
    def __init__(self, d_model=80, scales=(1, 2, 4, 8), target_token_len=64):
        super().__init__()
        self.target_token_len = target_token_len
        self.conv = ScaleEquivariantConv1d(1, d_model, 5, scales, stride=1)
        _ng = min(8, d_model)
        while d_model % _ng != 0 and _ng > 1:
            _ng -= 1
        self.mix = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.GroupNorm(_ng, d_model), nn.GELU(),
        )
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_raw):
        f = self.conv(x_raw.unsqueeze(1))
        f = F.adaptive_avg_pool1d(f, self.target_token_len)
        f = self.mix(f)
        return self.norm(self.proj(f.transpose(1, 2)))


# =============================================================
# 4. Lightweight complex equalizer
# =============================================================
class LightweightEqualizer(nn.Module):
    def __init__(self, d_model, taps=9):
        super().__init__()
        self.taps = taps
        self.filter_re = nn.Conv1d(d_model, d_model, taps,
                                    padding=taps // 2, groups=d_model, bias=False)
        self.filter_im = nn.Conv1d(d_model, d_model, taps,
                                    padding=taps // 2, groups=d_model, bias=False)
        nn.init.dirac_(self.filter_re.weight)
        nn.init.zeros_(self.filter_im.weight)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_i, x_q):
        xi, xq = x_i.transpose(1, 2), x_q.transpose(1, 2)
        yi = self.filter_re(xi) - self.filter_im(xq)
        yq = self.filter_re(xq) + self.filter_im(xi)
        yi, yq = yi.transpose(1, 2), yq.transpose(1, 2)
        g = torch.sigmoid(self.gate)
        return x_i + g * (yi - x_i), x_q + g * (yq - x_q)


# =============================================================
# 5. Transformer machinery
# =============================================================
class AdaptiveSEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.spatial_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.GELU(),
            nn.Linear(channels // reduction, 1), nn.Sigmoid())
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.fc1(x.mean(1) + x.max(1)[0]))
        return x * torch.sigmoid(self.fc2(y)).unsqueeze(1) * self.spatial_fc(x)


class LocalFeatureEnhancement(nn.Module):
    def __init__(self, d_model, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size,
                                padding=kernel_size // 2, groups=d_model, bias=False)
        self.pwconv = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        r = x; x = x.transpose(1, 2)
        x = self.pwconv(self.dwconv(x)).transpose(1, 2)
        return self.act(self.norm(x)) + r


class EfficientCMHSA(nn.Module):
    def __init__(self, d_model, n_head, seq_length, dropout=0.1, use_rpe=True):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.use_rpe = use_rpe
        self.max_seq = seq_length
        if use_rpe:
            self.rpe_table = nn.Parameter(torch.zeros(n_head, 2 * seq_length - 1))
            coords = torch.arange(seq_length)
            rel = coords.unsqueeze(0) - coords.unsqueeze(1) + seq_length - 1
            self.register_buffer("rel_idx", rel)
            trunc_normal_(self.rpe_table, std=0.02)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _rpe(self, L):
        if not self.use_rpe or L > self.max_seq:
            return 0
        idx = self.rel_idx[:L, :L]
        return self.rpe_table[:, idx.reshape(-1)].view(self.n_head, L, L)

    def forward(self, x_i, x_q):
        B, L, D = x_i.shape
        qkv_i = self.qkv(x_i).reshape(B, L, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i[0], qkv_i[1], qkv_i[2]
        qkv_q = self.qkv(x_q).reshape(B, L, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q_q, k_q, v_q = qkv_q[0], qkv_q[1], qkv_q[2]
        a_re = (q_i @ k_i.transpose(-2, -1) + q_q @ k_q.transpose(-2, -1)) * self.scale
        a_im = (q_q @ k_i.transpose(-2, -1) - q_i @ k_q.transpose(-2, -1)) * self.scale
        rpe = self._rpe(L); a_re = a_re + rpe; a_im = a_im + rpe
        a_re = self.attn_drop(F.softmax(a_re, dim=-1))
        a_im = self.attn_drop(F.softmax(a_im, dim=-1))
        out_i = (a_re @ v_i - a_im @ v_q).transpose(1, 2).reshape(B, L, D)
        out_q = (a_re @ v_q + a_im @ v_i).transpose(1, 2).reshape(B, L, D)
        return self.proj_drop(self.proj(out_i)), self.proj_drop(self.proj(out_q))


class DynamicIQFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cross_i = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.cross_q = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.gate_i = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_q = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model), nn.Sigmoid())
        self.n_i = nn.LayerNorm(d_model)
        self.n_q = nn.LayerNorm(d_model)
        self.n_f = nn.LayerNorm(d_model)

    def forward(self, x_i, x_q):
        a_i, _ = self.cross_i(x_i, x_q, x_q)
        a_q, _ = self.cross_q(x_q, x_i, x_i)
        x_i = self.n_i(x_i + a_i); x_q = self.n_q(x_q + a_q)
        c = torch.cat([x_i, x_q], dim=-1)
        g_i = self.gate_i(c); g_q = self.gate_q(c)
        return self.n_f(x_i + g_i * x_q), self.n_f(x_q + g_q * x_i)


class GatedFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.se = AdaptiveSEBlock(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        r = x
        a, b = self.fc1(x).chunk(2, dim=-1)
        x = self.fc2(self.drop(a * self.act(b)))
        return self.norm(self.se(x) + 0.1 * r)


class ComplexTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, seq_length,
                 dropout=0.1, drop_path=0.1, use_rpe=True, eq_taps=9):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.lfe = LocalFeatureEnhancement(d_model)
        self.attn = EfficientCMHSA(d_model, n_head, seq_length, dropout, use_rpe)
        self.eq = LightweightEqualizer(d_model, eq_taps)
        self.fuse = DynamicIQFusion(d_model)
        self.ffn_i = GatedFFN(d_model, d_ff, dropout)
        self.ffn_q = GatedFFN(d_model, d_ff, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x_i, x_q):
        x_i = self.lfe(x_i); x_q = self.lfe(x_q)
        x_i, x_q = self.eq(x_i, x_q)
        ai, aq = self.attn(self.norm1(x_i), self.norm1(x_q))
        x_i = x_i + self.drop_path(ai); x_q = x_q + self.drop_path(aq)
        x_i, x_q = self.fuse(x_i, x_q)
        x_i = x_i + self.drop_path(self.ffn_i(self.norm2(x_i)))
        x_q = x_q + self.drop_path(self.ffn_q(self.norm3(x_q)))
        return x_i, x_q


class DiscriminativeClassifier(nn.Module):
    def __init__(self, d_model, num_class, dropout=0.1):
        super().__init__()
        self.attn_i = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))
        self.attn_q = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))
        feat_dim = d_model * 10
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model * 4), nn.LayerNorm(d_model * 4),
            nn.GELU(), nn.Dropout(dropout))
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.LayerNorm(d_model * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.head = nn.Linear(d_model, num_class)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, cls_i, cls_q, tok_i, tok_q):
        ai = (tok_i * self.attn_i(tok_i)).sum(1)
        aq = (tok_q * self.attn_q(tok_q)).sum(1)
        avg_i, avg_q = tok_i.mean(1), tok_q.mean(1)
        max_i, max_q = tok_i.max(1)[0], tok_q.max(1)[0]
        std_i, std_q = tok_i.std(1), tok_q.std(1)
        f = torch.cat([cls_i, cls_q, ai, aq, avg_i, avg_q,
                       max_i, max_q, std_i, std_q], dim=-1)
        return self.head(self.bottleneck(self.proj(f)))


# =============================================================
# 6. Physics-aware augmentation
# =============================================================
class PhysicsAwareAugmentation(nn.Module):
    def __init__(self, p_phase=0.5, p_amp=0.3, p_noise=0.3, warmup_steps=1500):
        super().__init__()
        self.p_phase = p_phase
        self.p_amp = p_amp
        self.p_noise = p_noise
        self.warmup_steps = warmup_steps
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    def _warmup_factor(self):
        if self.warmup_steps <= 0:
            return 1.0
        return min(self._step.item() / self.warmup_steps, 1.0)

    def forward(self, x_i, x_q):
        if not self.training:
            return x_i, x_q
        self._step += 1
        w = self._warmup_factor()
        B = x_i.shape[0]; dev = x_i.device

        if torch.rand(1, device=dev).item() < self.p_phase * w:
            ph = 2 * math.pi * torch.rand(B, 1, device=dev)
            c, s = torch.cos(ph), torch.sin(ph)
            x_i, x_q = x_i * c - x_q * s, x_i * s + x_q * c

        if torch.rand(1, device=dev).item() < self.p_amp * w:
            a = 0.7 + 0.6 * torch.rand(B, 1, device=dev)
            x_i, x_q = x_i * a, x_q * a

        if torch.rand(1, device=dev).item() < self.p_noise * w:
            sp = (x_i.pow(2) + x_q.pow(2)).mean(-1, keepdim=True)
            snr = 10 + 20 * torch.rand(B, 1, device=dev)
            sig = torch.sqrt(sp / (10 ** (snr / 10)) / 2)
            x_i = x_i + torch.randn_like(x_i) * sig
            x_q = x_q + torch.randn_like(x_q) * sig

        return x_i, x_q


# =============================================================
# 7. Full POET model
# =============================================================
class POET(nn.Module):
    """
    Fully-supervised modulation classification network.

    Pipeline (frontend order matters):
        PhysicsAwareAugmentation   [phase / amp / noise; train only]
      → InstanceAGC                [power normalization]
      → ScaleEquivariantEmbedding  [multi-scale conv + cross-scale attention]
      → ComplexTransformerBlocks
      → DiscriminativeClassifier
    """
    def __init__(self,
                 d_model=80, n_head=4, d_ff=320,
                 seq_length=128, layer_num=3, num_class=11,
                 dropout=0.15, drop_path=0.15,
                 use_rpe=True,
                 scales=(1, 2, 4, 8),
                 eq_taps=9,
                 aug_warmup_steps=1500):
        super().__init__()
        self.d_model = d_model

        # --- Frontend
        self.aug = PhysicsAwareAugmentation(warmup_steps=aug_warmup_steps)
        self.agc = InstanceAGC()

        # --- Feature extraction
        target_token_len = seq_length // 2
        self.embed = ScaleEquivariantEmbedding(
            d_model=d_model, scales=scales, target_token_len=target_token_len)

        # --- Transformer + classifier
        max_tokens = target_token_len + 4
        total_len = max_tokens + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_len, d_model))
        self.input_norm = nn.LayerNorm(d_model)

        dpr = [x.item() for x in torch.linspace(0, drop_path, layer_num)]
        self.layers = nn.ModuleList([
            ComplexTransformerBlock(
                d_model, n_head, d_ff, total_len, dropout, dpr[i],
                use_rpe, eq_taps)
            for i in range(layer_num)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = DiscriminativeClassifier(d_model, num_class, dropout)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        _protected = set()
        if self.agc.gain_net is not None:
            _protected.add(id(self.agc.gain_net[-1].weight))
            _protected.add(id(self.agc.gain_net[-1].bias))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if id(m.weight) in _protected:
                    continue
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    if id(m.bias) in _protected:
                        continue
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                if id(m.weight) in _protected:
                    continue
                if m.weight.dim() == 3 and m.kernel_size[0] > 1:
                    if not torch.allclose(m.weight.sum(),
                                          torch.tensor(float(m.weight.shape[0])),
                                          atol=1e-3):
                        try:
                            nn.init.kaiming_normal_(
                                m.weight, mode="fan_out", nonlinearity="relu")
                        except ValueError:
                            pass
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _frontend(self, x_i, x_q):
        x_i, x_q = self.aug(x_i, x_q)
        x_i, x_q = self.agc(x_i, x_q)
        return x_i, x_q

    def _encode(self, x_i, x_q):
        tok_i = self.embed(x_i)
        tok_q = self.embed(x_q)
        B = tok_i.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tok_i = torch.cat([cls, tok_i], dim=1)
        tok_q = torch.cat([cls, tok_q], dim=1)
        L = tok_i.shape[1]
        if L > self.pos_embed.shape[1]:
            pe = F.interpolate(self.pos_embed.transpose(1, 2), size=L,
                               mode="linear", align_corners=False).transpose(1, 2)
        else:
            pe = self.pos_embed[:, :L]
        tok_i = self.input_norm(tok_i + pe)
        tok_q = self.input_norm(tok_q + pe)
        for layer in self.layers:
            tok_i, tok_q = layer(tok_i, tok_q)
        return self.final_norm(tok_i), self.final_norm(tok_q)

    def forward(self, x, state=None):
        if x.dim() == 4:
            x = x.squeeze(1)
        x_i, x_q = x[..., 0], x[..., 1]
        x_i, x_q = self._frontend(x_i, x_q)
        tok_i, tok_q = self._encode(x_i, x_q)
        return self.classifier(tok_i[:, 0], tok_q[:, 0],
                               tok_i[:, 1:], tok_q[:, 1:])

    def compute_aux_loss(self, *args, **kwargs):
        return torch.zeros((), device=next(self.parameters()).device)

    def get_cross_scale_attn_stats(self):
        stats = {}
        for name, m in self.named_modules():
            if isinstance(m, CrossScaleAttention) and m._last_attn is not None:
                stats[name] = m._last_attn.detach().mean(dim=(0, 2)).cpu().tolist()
        return stats


# =============================================================
# 8. Configurations
# =============================================================
def get_poet_config_rml2016():
    """Default config for fully-supervised POET on RML2016."""
    return {
        "d_model": 80, "n_head": 4, "d_ff": 320,
        "seq_length": 128, "layer_num": 3, "num_class": 11,
        "dropout": 0.15, "drop_path": 0.15,
        "use_rpe": True,
        "scales": (1, 2, 4, 8, 16),
        "eq_taps": 9,
        "aug_warmup_steps": 1500,
    }


def get_poet_config_rml2018():
    cfg = get_poet_config_rml2016()
    cfg.update({"seq_length": 1024, "num_class": 24, "d_ff": 384})
    return cfg


def numParams(net):
    return sum(int(np.prod(p.size())) for p in net.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 项目接入层


def build(*, num_classes: int, samples, params: dict) -> nn.Module:
    """按目录参数构建模型（输入 ``(B, 2, N)``、输出 logits）。

    ``seq_length`` 必须等于数据集窗口长度（切分目标 token 数与该长度绑定），
    因此由 ``samples`` 传入而不是独立可调参数。
    """
    if samples is None:
        raise ValueError("模型 poet 的结构依赖窗口长度：请提供 samples"
                         "（训练链路会从数据集契约传入）")
    model = POET(d_model=int(params["d_model"]),
                 n_head=int(params["n_head"]),
                 d_ff=int(params["d_ff"]),
                 seq_length=int(samples),
                 layer_num=int(params["layer_num"]),
                 num_class=int(num_classes),
                 dropout=float(params["dropout"]),
                 drop_path=float(params["drop_path"]),
                 use_rpe=bool(params["use_rpe"]),
                 scales=tuple(int(scale) for scale in params["scales"]))
    from ._adapters import CHANNELS_LAST, adapt

    return adapt(model, CHANNELS_LAST)
