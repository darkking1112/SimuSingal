"""``amc_net`` 实现：``custom/AMC_Net.py`` 的维护副本（原文件不改）。

相对原文件的差异**只有三处**，模型结构与计算不变：

1. ``AdaCorrModule`` 里的复数算子（``torch.fft.fft`` / 复乘 / ``torch.fft.ifft``）改为
   **实数化的 DFT 矩阵**实现（cos/sin 表作为 buffer，随 ``.to(device)`` 走）：数学等价，
   与 FFT 的 float32 差异约 1e-5～1e-4（相对），换来 ONNX 可导出（FFT 复数算子在
   opset 17 的现有导出器下会失败）；
2. ``copy.deepcopy`` 换成 ``clone``（张量语义相同）；
3. 文件末尾追加 ``build()`` 工厂，并去掉原文件的 ``__main__`` 演示块。

结构与参数约束的唯一声明在 ``signal_analysis.algorithms.amc.ai_model.amc_net``。
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.in_c = in_channel
        self.out_c = out_channel

        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c)
        )

    def forward(self, x):
        """
        x: [batchsize, C, H, W]
        """
        x = self.conv_block(x)

        return x


class MultiScaleModule(nn.Module):
    def __init__(self, out_channel):
        super(MultiScaleModule, self).__init__()
        self.out_c = out_channel

        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 7)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )

    def forward(self, x):
        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        x = torch.cat([y1, y2, y3], dim=1)

        return x


class TinyMLP(nn.Module):
    def __init__(self, N):
        super(TinyMLP, self).__init__()
        self.N = N

        self.mlp = nn.Sequential(
            nn.Linear(self.N, self.N // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.N // 4, self.N),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class AdaCorrModule(nn.Module):
    """自适应相关模块：用 DFT 矩阵做实数化实现（等价于 fft → 复乘 → ifft 取实部）。"""

    def __init__(self, N):
        super(AdaCorrModule, self).__init__()
        self.Im = TinyMLP(N)
        self.Re = TinyMLP(N)

        index = torch.arange(N, dtype=torch.float32)
        angle = 2 * math.pi * index[:, None] * index[None, :] / N
        # cos_table[k, n] = cos(2πkn/N)、sin_table[k, n] = sin(2πkn/N)
        self.register_buffer('cos_table', torch.cos(angle))
        self.register_buffer('sin_table', torch.sin(angle))

    def forward(self, x):
        # x:[B, C_out, 1, W]（W = 窗口长度）
        x_init = x.clone()
        # 实部/虚部：X_re[k] = Σ_n x[n]·cos(2πkn/W)，X_im[k] = -Σ_n x[n]·sin(2πkn/W)
        X_re = x @ self.cos_table.t()
        X_im = -(x @ self.sin_table.t())
        h_re = self.Re(X_re)
        h_im = self.Im(X_im)
        # 复乘（实部 h_re·X_re、虚部 h_im·X_im）后取逆变换的实部
        Y_re = h_re * X_re
        Y_im = h_im * X_im
        x = (Y_re @ self.cos_table - Y_im @ self.sin_table) / self.cos_table.shape[0]

        return x + x_init


class FeaFusionModule(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(FeaFusionModule, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value_heads)
        shape = context.size()
        context = context.contiguous().view(shape[0], -1, shape[-1])
        return context


class AMC_Net(nn.Module):
    def __init__(self,
                 num_classes=26,
                 sig_len=1024,
                 extend_channel=36,
                 latent_dim=512,
                 num_heads=2,
                 conv_chan_list=None):
        super(AMC_Net, self).__init__()
        self.sig_len = sig_len
        self.extend_channel = extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = conv_chan_list

        if self.conv_chan_list is None:
            self.conv_chan_list = [36, 64, 128, 256]
        self.stem_layers_num = len(self.conv_chan_list) - 1

        self.ACM = AdaCorrModule(self.sig_len)
        self.MSM = MultiScaleModule(self.extend_channel)
        self.FFM = FeaFusionModule(self.num_heads, self.sig_len, self.sig_len)

        self.Conv_stem = nn.Sequential()

        for t in range(0, self.stem_layers_num):
            self.Conv_stem.add_module(f'conv_stem_{t}',
                                      Conv_Block(
                                          self.conv_chan_list[t],
                                          self.conv_chan_list[t + 1])
                                      )

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes)
        )

    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = x.permute(0, 2, 1) 
        x = x.unsqueeze(1)
        x = self.ACM(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = self.MSM(x)
        x = self.Conv_stem(x)
        x = self.FFM(x.squeeze(2))
        x = self.GAP(x)
        y = self.classifier(x.squeeze(2))
        return y


# ---------------------------------------------------------------- 项目接入层


def build(*, num_classes: int, samples, params: dict) -> nn.Module:
    """按目录参数构建模型（输入 ``(B, 2, N)``、输出 logits）。

    ``sig_len`` 必须等于数据集窗口长度（AdaCorr 的 DFT 表、FFM 的注意力维度都与之绑定），
    因此由 ``samples`` 传入。原实现的 ``latent_dim`` 由 ``num_heads × conv_chan_list[-1]``
    决定（默认 2×256=512），不暴露成可调参数以免与结构不匹配。
    """
    if samples is None:
        raise ValueError("模型 amc_net 的结构依赖窗口长度：请提供 samples"
                         "（训练链路会从数据集契约传入）")
    channels = [int(width) for width in params["conv_chan_list"]]
    extend_channel = int(params["extend_channel"])
    num_heads = int(params["num_heads"])
    if channels[0] != extend_channel:
        raise ValueError(f"conv_chan_list 的第 1 个宽度应等于 extend_channel"
                         f"（{extend_channel}），当前是 {channels[0]}")
    if extend_channel % 3:
        raise ValueError("extend_channel 应能被 3 整除（多尺度分支各占 1/3）")
    if int(samples) % num_heads:
        raise ValueError(f"窗口长度 {int(samples)} 应能被注意力头数 {num_heads} 整除")
    model = AMC_Net(num_classes=int(num_classes), sig_len=int(samples),
                    extend_channel=extend_channel,
                    latent_dim=num_heads * channels[-1],
                    num_heads=num_heads, conv_chan_list=channels)
    from ._adapters import CHANNELS_LAST, adapt

    return adapt(model, CHANNELS_LAST)
