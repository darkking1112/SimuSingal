import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        # Keras 'causal' padding 意味着只在时间维度的左侧（过去）进行填充
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        # x shape: (Batch, Channel, Time)
        # F.pad 参数格式为 (left, right, top, bottom, ...)
        # 我们只在最后一个维度（Time）的左侧填充
        x = F.pad(x, (self.pad_len, 0))
        return self.conv(x)

class MCLDNN(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.5):
        super(MCLDNN, self).__init__()
        
        # Part-A: Spatial Characteristics Mapping
        
        # Path 1: Conv2D on combined I/Q (Input1 in Keras)
        # Keras: Conv2D(50, (2, 8), padding='same')
        # PyTorch Conv2d padding='same' 保持输入输出高宽一致 (需要 stride=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same'),
            nn.ReLU()
        )

        # Path 2 & 3: Conv1D on I and Q separately (Input2/3 in Keras)
        self.conv2 = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=50, kernel_size=8),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=50, kernel_size=8),
            nn.ReLU()
        )

        # Conv4: Processing concatenated I/Q 1D features
        # Keras: Conv2D(50, (1, 8), padding='same')
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU()
        )

        # Conv5: Merging Path 1 and Path 2/3
        # Keras: Conv2D(100, (2, 5), padding='valid')
        # Input channels = 50 (from conv1) + 50 (from conv4) = 100
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid'), # Valid padding
            nn.ReLU()
        )

        # Part-B: Temporal Characteristics (LSTM)
        # Conv5 output shape: (Batch, 100, 1, 124) -> Transpose to (Batch, 124, 100)
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, num_layers=1)

        # DNN
        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc3 = nn.Linear(128, num_classes)
        
        self.initialize_weight()

    def initialize_weight(self):
        # 模拟 Keras/TensorFlow 的初始化方式
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                # Keras default is glorot_uniform (Xavier Uniform)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # Keras LSTM: kernel=glorot_uniform, recurrent_kernel=orthogonal
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        # LSTM bias usually initialized to 0, or forget gate bias to 1
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # Input x shape 假设为: (Batch, 128, 2) 或 (Batch, 2, 128)
        # 我们统一转为 (Batch, Channels, Time) -> (Batch, 2, 128)
        if x.shape[1] == 128 and x.shape[2] == 2:
            x = x.permute(0, 2, 1)
        
        # Path 1 Input: (Batch, 1, 2, 128) - "Image" of I/Q
        x_img = x.unsqueeze(1) 
        x_1 = self.conv1(x_img) # Out: (Batch, 50, 2, 128)

        # Path 2 & 3 Inputs: Separated I and Q
        # x[:, 0:1, :] shape is (Batch, 1, 128)
        x_i = self.conv2(x[:, 0:1, :]) # Out: (Batch, 50, 128)
        x_q = self.conv3(x[:, 1:2, :]) # Out: (Batch, 50, 128)

        # Stack I and Q features to form a "2-row" feature map
        # Keras Reshape+Concat logic results in (Batch, 2, 128, 50) channels last
        # PyTorch Equivalent: Stack on height dimension (dim 2)
        x_23 = torch.stack([x_i, x_q], dim=2) # Out: (Batch, 50, 2, 128)
        
        x_23 = self.conv4(x_23) # Out: (Batch, 50, 2, 128)

        # Concatenate Path 1 and Path 2/3 outputs
        # Keras concat axis=-1 (channels)
        # PyTorch concat axis=1 (channels)
        x_all = torch.cat([x_1, x_23], dim=1) # Out: (Batch, 100, 2, 128)

        # Conv5
        x_all = self.conv5(x_all) # Out: (Batch, 100, 1, 124) (Valid padding reduces 128->124, 2->1)

        # Prepare for LSTM
        # Remove height dim (which is 1) and permute to (Batch, Time, Channels)
        x_all = x_all.squeeze(2) # (Batch, 100, 124)
        x_all = x_all.permute(0, 2, 1) # (Batch, 124, 100)

        # LSTM Layers
        # LSTM1 returns sequences (all time steps)
        x_lstm, _ = self.lstm1(x_all) 
        # LSTM2 returns last state (equivalent to return_sequences=False in Keras, but checking logic)
        # Keras code: LSTM2(128, name="LSTM2")(x) -> default return_sequences=False
        x_lstm, _ = self.lstm2(x_lstm) 
        
        # Take the last time step output
        x_last = x_lstm[:, -1, :] # (Batch, 128)

        # Dense Layers
        out = self.fc1(x_last)
        out = self.fc2(out)
        logits = self.fc3(out)

        return logits

if __name__ == '__main__':
    # 假设输入为 (Batch, Time, Channels) = (4, 128, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MCLDNN(num_classes=11).to(device)
    
    x = torch.rand((4, 128, 2)).to(device)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape) # Should be (4, 11)