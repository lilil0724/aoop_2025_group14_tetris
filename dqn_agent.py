import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    """我們的神經網路，輸入是狀態特徵，輸出一維的 Q-value"""
    def __init__(self, state_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1) # 輸出只有一個 Q-value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, learning_rate=0.0005, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        self.state_size = state_size
        self.gamma = gamma  # 折扣因子，對未來獎勵的重視程度
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 我們需要兩個網路：一個用於決策(policy_net)，一個用於計算目標Q值(target_net)
        # 這可以讓訓練更穩定
        self.policy_net = QNetwork(state_size).to(self.device)
        self.target_net = QNetwork(state_size).to(self.device)
        self._update_target_network()
        self.target_net.eval() # Target net 不進行訓練

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # 使用均方誤差作為損失函數
        
        # 經驗回放池
        self.memory = deque(maxlen=25000)

    def _update_target_network(self):
        """複製 policy_net 的權重到 target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, possible_states):
        """
        根據 ε-greedy 策略選擇最佳動作
        possible_states: 一個字典 {action: state_features, ...}
        """
        if not possible_states:
            return None

        # Epsilon-Greedy：以 ε 的機率隨機探索，(1-ε) 的機率選擇最佳動作
        if random.random() < self.epsilon:
            return random.choice(list(possible_states.keys()))
        else:
            # 利用網路預測，選擇 Q-value 最高的動作
            best_action = None
            max_q_value = -float('inf')
            
            with torch.no_grad():
                for action, state in possible_states.items():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_value = self.policy_net(state_tensor).item()
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_action = action
            return best_action
            
    def remember(self, state, action, reward, next_state, done):
        """將經驗存入回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """從回放池中取樣學習"""
        if len(self.memory) < batch_size:
            return 0 # 記憶體不足，暫不學習

        minibatch = random.sample(self.memory, batch_size)
        
        # 將數據轉換成 Torch Tensors
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = [m[1] for m in minibatch] # action 比較複雜，稍後處理
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = {action: torch.FloatTensor(np.array([m[3][action] for m in minibatch if action in m[3]])).to(self.device)
                       for action in set(a for m in minibatch for a in m[3])}
        dones = torch.BoolTensor([m[4] for m in minibatch]).to(self.device)
        
        # --- 計算 Loss ---
        # 1. 計算當前 state 的 Q-value (Q(s,a))
        # 這裡需要找到對應 action 的 state 特徵
        q_eval_list = []
        for i, m in enumerate(minibatch):
            state_tensor = torch.FloatTensor(m[0]).unsqueeze(0).to(self.device)
            q_eval_list.append(self.policy_net(state_tensor))
        q_eval = torch.cat(q_eval_list)

        # 2. 計算目標 Q-value (y)
        # y = reward + gamma * max_a'( Q_target(s', a') )
        next_q_values = torch.zeros(batch_size, device=self.device)
        if next_states: # 如果有下一步狀態
            with torch.no_grad():
                all_next_states_list = []
                # 處理 next_states 批次化
                for i, m in enumerate(minibatch):
                    if not m[4]: # if not done
                        possible_next_states = m[3]
                        if possible_next_states:
                            state_tensors = torch.FloatTensor(np.array(list(possible_next_states.values()))).to(self.device)
                            next_q_values[i] = self.target_net(state_tensors).max(0)[0]
        
        q_target = rewards + self.gamma * next_q_values * (~dones)
        q_target = q_target.unsqueeze(1)

        # 3. 計算損失
        loss = self.criterion(q_eval, q_target)

        # 4. 反向傳播與優化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰減 Epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
