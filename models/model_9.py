import torch
import torch.nn as nn
import torch.nn.functional as F


class UltimusBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dims):
        super(UltimusBlock, self).__init__()
        self.fc_k = nn.Linear(in_channels, out_channels)
        self.fc_q = nn.Linear(in_channels, out_channels)
        self.fc_v = nn.Linear(in_channels, out_channels)
        self.dims = dims
        self.out_fc = nn.Linear(out_channels, in_channels)
        
    def forward(self, x):
        batch_size, channels = x.shape[0], x.shape[1]
        
        # Fully connected layers K, Q, and V
        k = self.fc_k(x.view(batch_size, channels))
        q = self.fc_q(x.view(batch_size, channels))
        v = self.fc_v(x.view(batch_size, channels))
#         print(k.shape, q.shape, v.shape)

        # Attention
        scores = torch.matmul(q.t(), k) / (self.dims ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        z = torch.matmul(v, attention_weights)
        
        # Output FC layer
        z = self.out_fc(z.view(batch_size, -1))
        return z


class Net(nn.Module):
    def __init__(self, d_k):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.gap = nn.AvgPool2d(kernel_size=(32,32))
        self.ultimus1 = UltimusBlock(48, 8, d_k)
        self.ultimus2 = UltimusBlock(48, 8, d_k)
        self.ultimus3 = UltimusBlock(48, 8, d_k)
        self.ultimus4 = UltimusBlock(48, 8, d_k)
        
        self.fc_out = nn.Linear(48, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
#         print(x.shape)
        x = self.gap(x)
#         print(x.shape)
        x = self.ultimus1(x)
#         print(x.shape)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        x = self.fc_out(x)
        return x