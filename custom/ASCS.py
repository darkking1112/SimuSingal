import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 选择使用GPU（如果可用），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================ maxpool_samepadding ============================

def get_same_padding(kernel_size, stride):
    # 计算有效卷积核尺寸
    kernel_size_effective_h = kernel_size[0] + (kernel_size[0] - 1) * (stride[0] - 1)
    kernel_size_effective_w = kernel_size[1] + (kernel_size[1] - 1) * (stride[1] - 1)

    # 计算需要的填充
    pad_total_h = kernel_size_effective_h - 1
    pad_total_w = kernel_size_effective_w - 1

    pad_beg_h = pad_total_h // 2
    pad_end_h = pad_total_h - pad_beg_h
    pad_beg_w = pad_total_w // 2
    pad_end_w = pad_total_w - pad_beg_w

    return (pad_beg_h, pad_end_h, pad_beg_w, pad_end_w)


class SamePaddingMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SamePaddingMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = self.calculate_same_padding(kernel_size, stride)

    def calculate_same_padding(self, kernel_size, stride):
        # 计算有效卷积核尺寸
        kernel_size_effective_h = kernel_size[0] + (kernel_size[0] - 1) * (stride[0] - 1)
        kernel_size_effective_w = kernel_size[1] + (kernel_size[1] - 1) * (stride[1] - 1)

        # 计算需要的填充
        pad_total_h = kernel_size_effective_h - 1
        pad_total_w = kernel_size_effective_w - 1

        pad_beg_h = pad_total_h // 2
        pad_end_h = pad_total_h - pad_beg_h
        pad_beg_w = pad_total_w // 2
        pad_end_w = pad_total_w - pad_beg_w

        return (pad_beg_h, pad_end_h, pad_beg_w, pad_end_w)

    def forward(self, x):
        # 在输入张量上应用padding
        x = F.pad(x, (self.padding[2], self.padding[3], self.padding[0], self.padding[1]))
        # 应用最大池化
        return F.max_pool2d(x, self.kernel_size, self.stride)


# ============================ Noise_Reduction_Model ============================

class SELayer(nn.Module):
    #通过全局平均池化和两个全连接层来学习通道之间的依赖关系
    #使用sigmoid函数来生成通道权重
    def __init__(self, channel, reduction):  #reduction是降维的比例 channel是通道数
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),              #直接修改输入张量的值
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):           #定义了数据通过SELayer时的前向传播过程
        b, c, _, _ = x.size()       #通过self.avg_pool对输入特征图x进行全局平均池化，然后通过view方法将结果重塑为(b, c)的形状，其中b是批量大小，c是通道数
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  #原始特征图的形状是 [b, c, h, w]，我们需要将通道权重的形状也变为 [b, c, h, w]，以便进行元素级的乘法操作
        #print(y.expand_as(x).shape)  #y.expand_as(x)  实际上就是每个通道的软阈值
        return x * y.expand_as(x), y.expand_as(x)  #将生成的通道权重y与原始输入特征图x相乘，实现通道权重的重新校准。y.expand_as(x)确保权重的形状与x相同，以便进行元素级乘法


class SEBlock(nn.Module):
    #定义了一个包含SE层的卷积块，其中卷积层用于提取特征，SE层用于重新校准通道权重
    def __init__(self, channel, reduction, beta):
        super(SEBlock, self).__init__()
        self.beta = beta
        self.conv = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1))  #卷积核大小为3，填充为0
        self.bn = nn.BatchNorm2d(channel)
        self.se = SELayer(channel, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.abs(x)   #计算每个通道的绝对值
        x = F.relu(x)
        x, tau_threshold_vector = self.se(x)              #学习通道之间的依赖关系并重新校准通道权重
        soft_thresholded_x = self.soft_threshold(x, tau_threshold_vector)
        return soft_thresholded_x

    def soft_threshold(self, x, tau_threshold_vector):
        return torch.sign(x) * torch.max(torch.abs(x) - self.beta * tau_threshold_vector *
                                         torch.exp((1 - self.beta) * (tau_threshold_vector - torch.abs(x))), torch.zeros_like(x))


class ASSM(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super(ASSM, self).__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)   #直接修改输入张量的值
        self.conv2 = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 初始卷积层操作
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 计算空间软阈值向量
        abs_x = torch.abs(x)
        #计算每个空间位置在通道维度上的平均值
        #参数dim=1指定了沿着通道维度（dim 1）进行操作，而keepdim=True确保输出张量在该维度上保持原有的维度数
        avg_pooled = torch.mean(abs_x, dim=1, keepdim=True)
        conv1_result = self.conv2(avg_pooled)
        conv2_result = self.conv3(conv1_result)
        conv2_result = self.sigmoid(conv2_result)
        spatial_soft_threshold_vector = conv2_result * avg_pooled
        if self.alpha > 0:
            # 空间软阈值操作
            soft_thresholded_x = self.soft_threshold(x, spatial_soft_threshold_vector)
            # 恢复潜在信息损失
            recovered_x = self.conv4(soft_thresholded_x)
        elif self.alpha == 0:
            soft_thresholded_x = x * spatial_soft_threshold_vector.expand_as(x)
            recovered_x = soft_thresholded_x
        return recovered_x

    def soft_threshold(self, x, threshold_vector):
        return torch.sign(x) * torch.max(torch.abs(x) - self.alpha * threshold_vector *
                                         torch.exp((1 - self.alpha) * (threshold_vector - torch.abs(x))), torch.zeros_like(x))


class ASSE_Block(nn.Module):
    def __init__(self, in_channels, out_channels, alpha, beta, reduction):
        super(ASSE_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.assm_module = ASSM(in_channels=self.in_channels, out_channels=self.out_channels, alpha=self.alpha)
        self.se_block = SEBlock(channel=self.out_channels, reduction=self.reduction, beta=self.beta)

    def forward(self, x):
        x = self.assm_module(x)
        x = self.se_block(x)
        return x


# ============================ Inception_Residual_Block ============================

class CustomInceptionResidualBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(CustomInceptionResidualBlock, self).__init__()
        # Tower 1
        self.tower_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0),  # 适应128个输入通道
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels // 2, kernel_size=(2, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(outchannels // 2)
        )

        # Tower 2
        self.tower_2 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0),  # 适应128个输入通道
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels // 2, kernel_size=(2, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(outchannels // 2)
        )

        # Tower 3
        self.tower_3 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0),  # 适应128个输入通道
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels // 2, kernel_size=(2, 7)),
            nn.ReLU(),
            nn.BatchNorm2d(outchannels // 2)
        )

    def forward(self, x):
        # Compute the output for each tower
        tower_1_out = self.tower_1(x)
        tower_2_out = self.tower_2(x)
        tower_3_out = self.tower_3(x)
        # Concatenate the outputs along the channel dimension (axis=1)
        output = torch.cat((tower_1_out, tower_2_out, tower_3_out), dim=1)
        #output = x + output
        #output = F.relu(output)  #加上激活函数
        return output


# ============================ Residual_Block ============================

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #使用dropout效果不好，但是在加上FFM模块时，使用dropout 0.5验证集准确率能达到62！
        # 用于维度匹配的卷积层
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResidualNetworkBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_blocks=2):
        super(ResidualNetworkBlock, self).__init__()
        self.num_blocks = num_blocks
        self.residual_blocks = nn.ModuleList([
            BasicResidualBlock(in_channels if i == 0 else out_channels, out_channels, stride if i == 0 else 1)
            for i in range(num_blocks)
            #  列表推导式(list comprehension)的一种使用方式。它们建了一个 BasicResidualBlock 实例的列表，其中每个实例都是根据循环的当前迭代来配置的
        ])

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)
            #self.maxpool_22 = SamePaddingMaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        return x


# ============================ FFM_model ============================

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


# ============================ ClassifierNet ============================

class ClassifierNet(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierNet, self).__init__()
        # 全局平均池化层
        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 全连接层
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.PReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 全局平均池化
        x = self.global_avg_pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)  #增强网络的非线性表达能力
        x = self.fc2(x)
        # softmax激活函数用于获得概率分布
        #x = F.softmax(x, dim=1)
        return x


# ============================ ASCS ============================

def stft_sig(signal_tensor):
    I = signal_tensor[:, 0, :]
    Q = signal_tensor[:, 1, :]
    complex_signal = I + 1j * Q
    n_fft = 128 // 8
    hop_length = 1
    window = torch.hamming_window(n_fft).to(signal_tensor.device)
    stft_result = torch.stft(complex_signal, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    stft_result_real = stft_result.real.unsqueeze(1)
    stft_result_imag = stft_result.imag.unsqueeze(1)
    stft_result = torch.cat((stft_result_real, stft_result_imag), dim=1)

    return stft_result


class ASCS(nn.Module):
    def __init__(self, num_classes):

        super(ASCS, self).__init__()

        # ASSE模块，用于特征处理，输入通道为2，输出通道为16，reduction为4
        self.asse_block = ASSE_Block(
            in_channels=2,
            out_channels=16,
            reduction=4,
            alpha=0.6,
            beta=0.6
        )
        # 另一个ASSE模块，用于进一步处理特征，输入输出通道都为16，reduction为4
        self.asse_attention = ASSE_Block(
            in_channels=16,
            out_channels=16,
            reduction=4,
            alpha=0.6,
            beta=0.6
        )
        # 又一个ASSE模块，用于处理特征，输入输出通道都为96，reduction为16
        self.asse_attention1 = ASSE_Block(
            in_channels=96,
            out_channels=96,
            reduction=16,
            alpha=0.6,
            beta=0.6
        )

        self.CIRB_block = CustomInceptionResidualBlock(inchannels=16, outchannels=32)
        self.Residual_Block1 = ResidualNetworkBlock(in_channels=48, out_channels=96, stride=1, num_blocks=1)
        self.Residual_Block2 = ResidualNetworkBlock(in_channels=96, out_channels=128, stride=1, num_blocks=1)
        self.ffm_model = FeaFusionModule(num_attention_heads=2, input_size=72, hidden_size=128)  #72
        self.maxpool_22 = SamePaddingMaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.Classifier = ClassifierNet(num_classes)

    def forward(self, x):
        x = stft_sig(x)

        x = self.asse_block(x)
        x = self.asse_attention(x)
        x = self.maxpool_22(x)
        x = self.CIRB_block(x)
        x = self.maxpool_22(x)
        x = self.Residual_Block1(x)
        x = self.asse_attention1(x)
        x = self.Residual_Block2(x)
        x = self.maxpool_22(x)

        # 展平空间维度
        x = x.view(x.size(0), x.size(1), -1)
        x = self.ffm_model(x)
        x = self.Classifier(x)
        return x


if __name__ == '__main__':
    model = ASCS(num_classes=11).to(device)
    print(model)
