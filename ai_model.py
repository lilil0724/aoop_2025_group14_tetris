import torch
import torch.nn as nn

class TetrisActorCritic(nn.Module):
    def __init__(self, input_dim=4): # 輸入只有 4 個特徵
        super(TetrisActorCritic, self).__init__()
        
        # 簡單的全連接層 (MLP)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor (輸出喜好分數)
        self.actor_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        # Critic (輸出價值)
        self.critic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (Batch, 4)
        features = self.shared_net(x)
        return self.actor_head(features), self.critic_head(features)
