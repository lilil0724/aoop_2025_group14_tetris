import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from tqdm import tqdm
import os
import torch.nn as nn  # <--- 請確認加上這一行
from tetris_env import TetrisEnv
from ai_model import TetrisTransformer
import config

# --- Hyperparameters ---
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 2000
LR = 1e-4
MEMORY_SIZE = 20000
TARGET_UPDATE = 5
NUM_EPISODES = 300 # 建議至少跑 300-500 場

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, next_state, reward, done):
        # state: numpy array
        self.memory.append((state, action, next_state, reward, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def train():
    env = TetrisEnv()
    
    policy_net = TetrisTransformer(config.rows, config.columns).to(device)
    target_net = TetrisTransformer(config.rows, config.columns).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    steps_done = 0
    scores = []
    
    print(f"Start training on: {device}")
    
    # Tqdm 進度條
    progress_bar = tqdm(range(NUM_EPISODES))
    
    for i_episode in progress_bar:
        # 初始化
        env.reset()
        
        # 為了簡化，TetrisEnv.reset 不直接回傳 state，我們手動拿第一步
        # 但我們的 env 設計是 "選擇下一步最好的盤面"，所以 flow 稍微不同：
        # 1. 獲取所有可能盤面 -> 2. AI 評分 -> 3. 選最高的 -> 4. 執行 -> Loop
        
        total_reward = 0
        
        while True:
            # 1. 獲取當前所有可能的 "下一步狀態"
            # states dict: {(x, rot): tensor_numpy}
            next_states_dict = env.get_possible_next_states()
            
            if not next_states_dict:
                break # Game Over (無處可放)
            
            # 準備資料
            moves = list(next_states_dict.keys())
            states_np = list(next_states_dict.values())
            
            # 將 numpy list 轉成 Tensor Batch
            state_batch = torch.tensor(np.array(states_np), dtype=torch.float32).to(device)
            
            # 2. Epsilon-Greedy 策略
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            use_ai = True
            if random.random() < eps_threshold:
                use_ai = False
            
            if use_ai:
                with torch.no_grad():
                    # AI 給每個可能的 "未來盤面" 打分
                    q_values = policy_net(state_batch)
                    best_idx = q_values.argmax().item()
            else:
                # 隨機選一個
                best_idx = random.randint(0, len(moves) - 1)
            
            action = moves[best_idx]         # 選擇的動作 (x, rot)
            chosen_state = states_np[best_idx] # 選擇後的盤面狀態 (作為 Input)
            
            # 3. 執行動作
            reward, done = env.step(action)
            total_reward += reward
            
            # 4. 儲存記憶
            # 這裡有個 Trick: 在這種 "從所有可能狀態選一個" 的做法中
            # 我們通常把 "選擇後的那個盤面" 當作 State 存起來
            # Next State 則是下一輪 "最好的那個盤面" (但在這裡不好取得，我們先簡化存 reward)
            
            # 為了能跑 DQN，我們需要存: (Current_State_Img, Reward, Done)
            # 這裡簡化: 我們只訓練 AI 辨識 "好盤面"，所以把 (Chosen_State, Reward) 存進去
            # 這種變體有時稱為 Direct Policy Learning 或 Approximate Value Iteration
            
            if len(memory) > BATCH_SIZE:
                 transitions = memory.sample(BATCH_SIZE)
                 # 解壓縮
                 batch_states, _, _, batch_rewards, batch_dones = zip(*transitions)
                 
                 b_states = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
                 b_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device).unsqueeze(1)
                 b_dones = torch.tensor(batch_dones, dtype=torch.float32).to(device).unsqueeze(1)
                 
                 # 計算 Loss
                 # 目標: 預測的 Q 值應該接近 Reward (如果是 Done) 
                 # 或者 Reward + Gamma * Max_Next_Q (這裡暫時簡化為只看當前 Reward 以加速收斂測試)
                 
                 current_q = policy_net(b_states)
                 
                 # 簡化版 Loss: 讓 AI 預測的分數接近 Reward
                 # 進階版應該要算 Next State 的 Max Q，但在 Tetris 這種擁有大量分支的遊戲比較複雜
                 target_q = b_rewards 
                 
                 loss = nn.MSELoss()(current_q, target_q)
                 
                 optimizer.zero_grad()
                 loss.backward()
                 optimizer.step()
            
            # 將剛剛的經驗存入
            memory.push(chosen_state, action, None, reward, done)
            
            if done:
                break
        
        scores.append(total_reward)
        progress_bar.set_description(f"Ep {i_episode} | R: {total_reward:.1f} | Avg: {np.mean(scores[-10:]):.1f}")
        
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # 儲存模型
    torch.save(policy_net.state_dict(), "tetris_transformer.pth")
    print("Training Complete! Model saved as tetris_transformer.pth")

if __name__ == "__main__":
    train()

git config --global user.name "lilil0724"
git config --global user.email "kyle0724.ee12@nycu.edu.tw"