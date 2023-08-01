import torch
from torch import nn

########################################## TS-STDN ##########################################
# Name : Two-Stream Spectral-Temporal Denoising Network
# Time : 2023.06.30
# Author : Xuanhao Liu
# Affiliation : Shanghai Jiao Tong University
# Conference : ICONIP 2023
# The C, T, D, Fs, Ft, d is the same meaning as the paper says
# classes_num   : the number of categories
# C             : the number of channels
# T             : the time samples of EEG signals, such as 200 for a 1sec signals with 200Hz
# D             : the depth of the Depthwise Convolution
# Fs            : the filters' number of the Spectral Stream Feature Convolution layer
# Ft            : the filters' number of the Temporal Stream Feature Convolution layer
# d             : the embedding dimension of LSTM
#############################################################################################

class TS_STDN(nn.Module):
    def __init__(self, classes_num, C, T, D, Fs, Ft, d):
        super(TS_STDN, self).__init__()
        
        self.drop_out = 0.5
        self.Fa = Fs + Ft
            
        self.temporal_down1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,                  # input shape  (1, C, T)
                out_channels=2,                 # num_filters
                kernel_size=(1, 64),            # filter size
                bias=False
            ),                                  # output shape (2, C, T)
            nn.BatchNorm2d(2),                  # output shape (2, C, T)
            nn.ELU(), 
            nn.AvgPool2d((1, 2)),               # output shape (2, C, T//2)
        )
        
        self.temporal_down2 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(
                in_channels=2,                  # input shape  (2, C, T//2)
                out_channels=4,                 # num_filters
                kernel_size=(1, 32),            # filter size
                bias=False
            ),                                  # output shape (4, C, T//2)
            nn.BatchNorm2d(4),                  # output shape (4, C, T//2)
            nn.ELU(), 
            nn.AvgPool2d((1, 2)),               # output shape (4, C, T//4)
        )
        
        self.temporal_lowest = nn.Sequential(
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T//4)
                out_channels=4,                 # num_filters
                kernel_size=(1, 1),             # filter size
                groups=4,
                bias=False
            ), 
            nn.BatchNorm2d(4),
            nn.ELU(),                           # output shape (4, C, T//4)
        )
        
        self.temporal_up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2)),    # input shape  (4, C, T//4)
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T//2)
                out_channels=2,                 # num_filters
                kernel_size=(1, 32),            # filter size
                groups=2,
                bias=False
            ), 
            nn.BatchNorm2d(2),
            nn.ELU(),                           # output shape (2, C, T//2)
        )
        
        self.temporal_up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2)),    # input shape  (4, C, T//2)
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T)
                out_channels=2,                 # num_filters
                kernel_size=(1, 64),            # filter size
                groups=2,
                bias=False
            ), 
            nn.BatchNorm2d(2),
            nn.ELU(),                           # output shape (2, C, T) 
        )
        
        self.temporal_decoder = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=3,                  # input shape  (3, C, T)
                out_channels=1,                 # num_filters
                kernel_size=(1, 64),            # filter size
                bias=False
            ),
            nn.ReLU(),
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,                  # input shape  (1, C, T)
                out_channels=1,                 # num_filters
                kernel_size=(1, 64),            # filter size
                bias=False
            ),                                  # output shape (1, C, T) 
        )
        
        self.temporal_depthwise_CNN = nn.Sequential(
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T//4)
                out_channels=4 * D,             # num_filters
                kernel_size=(C, 1),             # filter size
                groups=4,
                bias=False
            ),                                  # output shape (4 * D, 1, T//4)
            nn.BatchNorm2d(4 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),               # output shape (4 * D, 1, T//8)
            nn.Dropout(self.drop_out)           # output shape (4 * D, 1, T//8)
        )
        
        self.temporal_feature_CNN = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=4 * D,              # input shape  (4 * D, 1, T//8)
                out_channels=Ft,                # num_filters
                kernel_size=(1, 16),            # filter size
                bias=False
            ),                                  # output shape (Ft, 1, T//8)
            nn.BatchNorm2d(Ft),                 # output shape (Ft, 1, T//8)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),               # output shape (Ft, 1, T//32)
            nn.Dropout(self.drop_out)           # output shape (Ft, 1, T//32)
        )
        
        self.spectral_down1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,                  # input shape  (1, C, T)
                out_channels=2,                 # num_filters
                kernel_size=(1, 64),            # filter size
                bias=False
            ),                                  # output shape (2, C, T)
            nn.BatchNorm2d(2),                  # output shape (2, C, T)
            nn.ELU(), 
            nn.AvgPool2d((1, 2)),               # output shape (2, C, T//2)
        )
        
        self.spectral_down2 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(
                in_channels=2,                  # input shape  (2, C, T//2)
                out_channels=4,                 # num_filters
                kernel_size=(1, 32),            # filter size
                bias=False
            ),                                  # output shape (4, C, T//2)
            nn.BatchNorm2d(4),                  # output shape (4, C, T//2)
            nn.ELU(), 
            nn.AvgPool2d((1, 2)),               # output shape (4, C, T//4)
        )
        
        self.spectral_lowest = nn.Sequential(
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T//4)
                out_channels=4,                 # num_filters
                kernel_size=(1, 1),             # filter size
                groups=4,
                bias=False
            ), 
            nn.BatchNorm2d(4),
            nn.ELU(),                           # output shape (4, C, T//4)
        )
        
        self.spectral_up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2)),    # input shape  (4, C, T//4)
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T//2)
                out_channels=2,                 # num_filters
                kernel_size=(1, 32),            # filter size
                groups=2,
                bias=False
            ), 
            nn.BatchNorm2d(2),
            nn.ELU(),                           # output shape (2, C, T//2)
        )
        
        self.spectral_up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2)),    # input shape  (4, C, T//2)
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T)
                out_channels=2,                 # num_filters
                kernel_size=(1, 64),            # filter size
                groups=2,
                bias=False
            ), 
            nn.BatchNorm2d(2),
            nn.ELU(),                           # output shape (2, C, T) 
        )
        
        self.spectral_decoder = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=3,                  # input shape  (3, C, T)
                out_channels=1,                 # num_filters
                kernel_size=(1, 64),            # filter size
                bias=False
            ),                                  # output shape (1, C, T)
            nn.ReLU(),
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,                  # input shape  (1, C, T)
                out_channels=1,                 # num_filters
                kernel_size=(1, 64),            # filter size
                bias=False
            ),                                  # output shape (1, C, T) 
        )
        
        self.spectral_depthwise_CNN = nn.Sequential(
            nn.Conv2d(
                in_channels=4,                  # input shape  (4, C, T//4)
                out_channels=4 * D,             # num_filters
                kernel_size=(C, 1),             # filter size
                groups=4,
                bias=False
            ),                                  # output shape (4 * D, 1, T//4)
            nn.BatchNorm2d(4 * D),              # output shape (4 * D, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),               # output shape (4 * D, 1, T//8)
            nn.Dropout(self.drop_out)           # output shape (4 * D, 1, T//8)
        )
        
        self.spectral_feature_CNN = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=4 * D,              # input shape  (4 * D, 1, T//8)
                out_channels=Fs,                # num_filters
                kernel_size=(1, 16),            # filter size
                bias=False
            ),                                  # output shape (Fs, 1, T//8)
            nn.BatchNorm2d(Fs),                 # output shape (F2, 1, T//8)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),               # output shape (F2, 1, T//32)
            nn.Dropout(self.drop_out)
        )
        
        self.lstm_embedding = nn.Linear(T//32, d)
        
        self.lstm = nn.LSTM(d, d, batch_first = True)
        
        self.attention_weight = nn.Sequential(
            nn.Linear(Fs+Ft, (Fs+Ft)//4),
            nn.Tanh(),
            nn.Linear((Fs+Ft)//4, Fs+Ft),
            nn.Softmax(dim=-1)
        )
        
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear((Fs+Ft)*d, classes_num)
        )
    
    def torch_stft(self, X_train):
        signal = []
        # index = np.arange(X_train.shape[1])
        # np.random.shuffle(index)
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 200,
                hop_length = 200,
                center = False,
                onesided = True)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=2)
    
    def forward(self, x):
        # x is the temporal data, y is the spectral data
        y = self.torch_stft(x.reshape(x.shape[0], x.shape[2], x.shape[3]))
        y = y.reshape(y.shape[0], 1 , y.shape[1], y.shape[2]*y.shape[3])
        
        down1 = self.temporal_down1(x)
        down2 = self.temporal_down2(down1)
        x_encoder = self.temporal_lowest(down2)
        up2 = self.temporal_up2(x_encoder)
        up2 = torch.cat([down1, up2], dim=1)
        up1 = self.temporal_up1(up2)
        up1 = torch.cat([x, up1], dim=1)
        x_denoise = self.temporal_decoder(up1)
        
        down1 = self.spectral_down1(y)
        down2 = self.spectral_down2(down1)
        y_encoder = self.spectral_lowest(down2)
        up2 = self.spectral_up2(y_encoder)
        up2 = torch.cat([down1, up2], dim=1)
        up1 = self.spectral_up1(up2)
        up1 = torch.cat([y, up1], dim=1)
        y_denoise = self.spectral_decoder(up1)
        
        x = self.temporal_depthwise_CNN(x_encoder)
        x = self.temporal_feature_CNN(x)
        
        y = self.spectral_depthwise_CNN(y_encoder)
        y = self.spectral_feature_CNN(y)
        
        xy = torch.cat([x, y], dim=1).reshape(x.shape[0], x.shape[1]*2, x.shape[3])
        
        xy = self.lstm_embedding(xy)
        xy, _ = self.lstm(xy)
        
        w = self.attention_weight(xy.mean(-1)).reshape(x.shape[0], self.Fa, 1)
        xy = torch.mul(w, xy)
        xy = self.out(xy)
        return xy, x_denoise, y_denoise

if __name__ == "__main__":
    model = TS_STDN(classes_num=3, C=62, T=400, D=4, d=16, Fs=16, Ft=16)
    x = torch.rand(size=(1, 1, 62, 400))
    print(x.shape)
    y, recons_t, recons_s = model(x)
    print(y.shape, recons_t.shape, recons_s.shape)