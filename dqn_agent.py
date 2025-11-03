# dqn_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from collections import deque

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class TetrisTransformer(nn.Module):
    def __init__(self, channels=2, rows=20, cols=10, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.conv = nn.Conv2d(channels, d_model, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=rows * cols)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(2).permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.output_layer(x)

class DQNAgent:
    def __init__(self, state_shape=(2, 20, 10), learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9999):
        self.state_shape = state_shape
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing agent on device: {self.device}")

        self.policy_net = TetrisTransformer(channels=state_shape[0]).to(self.device)
        self.target_net = TetrisTransformer(channels=state_shape[0]).to(self.device)
        
        self.reconfigure_for_phase({
            'learning_rate': learning_rate,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay
        })

        self._update_target_network()
        self.target_net.eval()

        self.criterion = nn.SmoothL1Loss()
        self.memory = deque(maxlen=50000)

    def _update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reconfigure_for_phase(self, phase_config):
        """(新增) 用於在每個新階段開始時重設超參數"""
        print("\n--- Reconfiguring Agent for New Phase ---")
        self.learning_rate = phase_config['learning_rate']
        self.epsilon = phase_config['epsilon_start']
        self.epsilon_start = phase_config['epsilon_start']
        self.epsilon_end = phase_config['epsilon_end']
        self.epsilon_decay = phase_config['epsilon_decay']
        
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)
        print(f"New Learning Rate: {self.learning_rate}")
        print(f"New Epsilon Range: {self.epsilon_start:.4f} -> {self.epsilon_end:.4f} (Decay: {self.epsilon_decay})")

    def act(self, possible_states):
        if not possible_states: return None
        if random.random() < self.epsilon: return random.choice(list(possible_states.keys()))

        with torch.no_grad():
            actions, states = zip(*possible_states.items())
            states_tensor = torch.from_numpy(np.stack(states)).float().to(self.device)
            q_values = self.policy_net(states_tensor)
            return actions[torch.argmax(q_values).item()]
            
    def remember(self, state, action, reward, next_possible_states, done):
        self.memory.append((state, action, reward, next_possible_states, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return 0

        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.from_numpy(np.stack([m[0] for m in minibatch])).float().to(self.device)
        current_q_values = self.policy_net(states)

        next_q_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            for i, m in enumerate(minibatch):
                if not m[4] and m[3]:
                    next_states_tensor = torch.from_numpy(np.stack(list(m[3].values()))).float().to(self.device)
                    next_q_values[i] = self.target_net(next_states_tensor).max()
        
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        dones = torch.BoolTensor([m[4] for m in minibatch]).to(self.device)
        target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_end: self.epsilon *= self.epsilon_decay
        return loss.item()
