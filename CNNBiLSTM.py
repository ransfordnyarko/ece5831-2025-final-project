
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTM_Model(nn.Module):
    def __init__(self, num_classes, n_mels=64):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # After two 2x2 pools: n_mels / 4
        self.lstm_input_size = 64 * (n_mels // 4)

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = self.cnn(x)
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, T, C, H)
        x = x.view(B, T, C * H)                  # (B, T, C*H)
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        return self.classifier(x)
