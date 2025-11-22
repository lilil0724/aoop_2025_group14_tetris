import torch
import torch.nn as nn
import math

class TetrisTransformer(nn.Module):
    def __init__(self, rows=20, cols=10, d_model=64, nhead=4, num_layers=2):
        super(TetrisTransformer, self).__init__()
        
        self.rows = rows
        self.cols = cols
        
        # 1. CNN 特徵提取器
        # 輸入: (Batch, 2, 20, 10) -> 通道0: 固定方塊, 通道1: 當前移動方塊
        self.embedding = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (16, 10, 5)
            nn.Conv2d(16, d_model, kernel_size=3, padding=1), # -> (64, 10, 5)
            nn.ReLU()
        )
        
        # 計算展平後的序列長度 (10 * 5 = 50)
        self.seq_len = (rows // 2) * (cols // 2) 
        
        # 2. 位置編碼 (Positional Encoding)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 價值輸出頭 (Value Head) - 評估這個盤面好不好 (DQN)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * self.seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 輸出單一分數 (Q-value)
        )

    def forward(self, x):
        # x shape: (Batch, 2, 20, 10)
        
        # CNN 處理 -> (Batch, 64, 10, 5)
        features = self.embedding(x)
        
        # 調整形狀給 Transformer: (Batch, 64, 50) -> (Batch, 50, 64)
        # Flatten height and width into sequence
        b, c, h, w = features.size()
        features = features.view(b, c, h * w).permute(0, 2, 1)
        
        # 加入位置編碼
        features = features + self.pos_encoder
        
        # Transformer 思考
        memory = self.transformer_encoder(features)
        
        # 展平: (Batch, 50 * 64)
        flat_memory = memory.reshape(memory.size(0), -1)
        
        # 輸出分數
        value = self.value_head(flat_memory)
        
        return value
