import time
import torch
import torch.nn as nn
import torch.nn.init as init

class PET(nn.Module):
    def __init__(self, frame_length=128):
        super(PET, self).__init__()
        self.p1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(frame_length * 2, 1)
            # Keras code uses 'linear' activation (default) for this Dense layer
        )

    def forward(self, x):
        # x shape: (Batch, Length, 2)
        
        # 1. Calculate Phase Theta
        # Output shape: (Batch, 1)
        p1_x = self.p1(x) 
        
        # 2. Sin/Cos Transformation
        # Expand dims to (Batch, 1, 1) for broadcasting against (Batch, Length, 1)
        sin_x = torch.sin(p1_x).unsqueeze(1) 
        cos_x = torch.cos(p1_x).unsqueeze(1)

        # 3. Separation of I and Q
        # x[:, :, 0] -> (Batch, Length) -> Unsqueeze to (Batch, Length, 1)
        x_i = x[:, :, 0:1] 
        x_q = x[:, :, 1:2] 

        # 4. Rotation Logic (Matching Keras Code)
        # Keras: x11 = input1(I) * cos, x12 = input2(Q) * sin
        # Keras: y1 = x11 + x12
        y1 = x_i * cos_x + x_q * sin_x

        # Keras: x21 = input2(Q) * cos, x22 = input1(I) * sin
        # Keras: y2 = x21 - x22
        y2 = x_q * cos_x - x_i * sin_x

        # 5. Concatenation and Reshape for Conv2D
        # Cat along last dim -> (Batch, Length, 2)
        x2 = torch.cat([y1, y2], dim=2)
        
        # Prepare for Conv2d (NCHW format)
        # N=Batch, C=1, H=Length, W=2
        x2 = x2.unsqueeze(1) 
        
        return x2

class PETCGDNN(nn.Module):
    def __init__(self, num_classes=26, frame_length=1024, hidden_size=128):
        super(PETCGDNN, self).__init__()
        
        # Backbone Feature Extractor
        self.pet = PET(frame_length=frame_length)
        
        # Conv Layer 1
        # Keras: Conv2D(75, (8,2), padding='valid') on input (128, 2, 1)
        # PyTorch Input: (Batch, 1, Length, 2) -> Kernel must be (8, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(8, 2), padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Conv Layer 2
        # Keras: Conv2D(25, (5,1), padding='valid')
        # Previous Output: (Batch, 75, Length-7, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(5, 1), padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Recurrent Layer
        # Keras: CuDNNGRU(units=128)
        self.gru = nn.GRU(input_size=25, hidden_size=hidden_size, batch_first=True)
        
        # Classifier
        # Keras: Dense(classes, activation='softmax')
        # Note: PyTorch CrossEntropyLoss includes Softmax, so we output linear logits here.
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights to match Keras 'glorot_uniform' (Xavier Uniform)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param) # Keras RNN default
                    elif 'bias' in name:
                        init.constant_(param, 0)

    def forward(self, x):
        # Input x: (Batch, 2, Length) or (Batch, Length, 2)
        # We enforce (Batch, Length, 2) for PET input
        if x.shape[1] == 2 and x.shape[2] != 2:
            x = x.permute(0, 2, 1)
            
        # 1. PET Block
        # Out: (Batch, 1, Length, 2)
        x = self.pet(x)
        
        # 2. Conv Blocks
        # Out: (Batch, 75, L-7, 1)
        x = self.conv1(x) 
        # Out: (Batch, 25, L-11, 1)
        x = self.conv2(x) 
        
        # 3. Reshape for GRU
        # Remove the width dimension (which is 1)
        x = x.squeeze(3) # (Batch, 25, L_new)
        # Permute to (Batch, Time, Features) for GRU
        x = x.permute(0, 2, 1) # (Batch, L_new, 25)
        
        # 4. GRU
        # x: (Batch, L_new, hidden_size)
        x, _ = self.gru(x)
        
        # Take the last time step
        x = x[:, -1, :]
        
        # 5. Classifier
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    FRAME_LEN = 128 
    
    model = PETCGDNN(num_classes=11, frame_length=FRAME_LEN).to(device)
    
    # Dummy input: (Batch, 2, 128)
    x = torch.rand((400, 2, FRAME_LEN)).to(device)
    
    # Warmup
    _ = model(x)
    
    # Timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    y = model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    print(f"Output Shape: {y.shape}")
    print(f"Time per sample: {(end-start)/400.0:.6f} s")