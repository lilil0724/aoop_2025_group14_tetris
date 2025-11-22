import torch
import torch.nn as nn
import numpy as np

class TetrisActorCritic(nn.Module):
    def __init__(self, rows=20, cols=10, d_model=64, nhead=4, num_layers=2):
        super(TetrisActorCritic, self).__init__()
        
        # --- 共用的感知層 (CNN + Transformer) ---
        # 這部分跟原本一樣，負責「看懂」盤面
        self.embedding = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(16, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.seq_len = (rows // 2) * (cols // 2) 
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 雙頭龍設計 ---
        
        # 1. 評論家 (Critic): 評估這個盤面「有多好」 (輸出 1 個分數)
        self.critic_head = nn.Sequential(
            nn.Linear(d_model * self.seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 2. 演員 (Actor): 決定「該選哪一個動作」
        # 注意: 因為 Tetris 的動作數量是不固定的 (取決於能放哪)，
        # 我們這裡稍微變通：我們不輸出固定的動作機率，
        # 而是輸出一個 "Preference Score"，搭配 Softmax 來選動作。
        self.actor_head = nn.Sequential(
            nn.Linear(d_model * self.seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 給當前輸入的 "這個狀態" 打一個 "喜好分"
        )

    def forward(self, x):
        # x: (Batch, 2, 20, 10)
        
        features = self.embedding(x)
        b, c, h, w = features.size()
        features = features.view(b, c, h * w).permute(0, 2, 1)
        features = features + self.pos_encoder
        memory = self.transformer_encoder(features)
        flat_memory = memory.reshape(memory.size(0), -1)
        
        value = self.critic_head(flat_memory) # 這裡有多好?
        policy_logits = self.actor_head(flat_memory) # 我有多想選這裡?
        
        return policy_logits, value
