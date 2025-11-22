import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
from tqdm import tqdm
import os
import math

from tetris_env import TetrisEnv
from ai_model import TetrisTransformer
import config

# --- 進階超參數設定 (Hyperparameters) ---
BATCH_SIZE = 128         # 批次大小
GAMMA = 0.99             # 折扣因子 (遠見)
EPS_START = 1.0          # 初始探索率 (100% 隨機)
EPS_END = 0.01           # 最終探索率 (1% 隨機)
EPS_DECAY = 5000         # 探索率衰減速度 (越慢越好)
LR = 1e-4                # 學習率
MEMORY_SIZE = 50000      # 記憶庫加大
TARGET_UPDATE = 10       # 每 10 場更新一次 Target Net
NUM_EPISODES = 10000     # 總場數: 1萬場 (建議掛機跑)
CHECKPOINT_FREQ = 500    # 每 500 場存一次檔

# --- 獎勵設定微調 (Reward Shaping) ---
# 這些可以在 tetris_env.py 裡修改，或者我們在這裡直接覆寫 env 的 step 邏輯 (如果結構允許)
# 這裡我們假設 tetris_env.py 已經定義好了基礎獎勵，我們主要靠大量訓練來克服

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def train():
    # 1. 初始化環境與模型
    env = TetrisEnv()
    
    policy_net = TetrisTransformer(config.rows, config.columns).to(device)
    target_net = TetrisTransformer(config.rows, config.columns).to(device)
    
    # 2. 檢查是否有存檔，實現「斷點續訓」
    start_episode = 0
    if os.path.exists("tetris_transformer_checkpoint.pth"):
        print("發現檢查點！正在載入繼續訓練...")
        checkpoint = torch.load("tetris_transformer_checkpoint.pth")
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(policy_net.state_dict())
        start_episode = checkpoint['episode']
        print(f"從第 {start_episode} 場繼續...")
    else:
        target_net.load_state_dict(policy_net.state_dict())
    
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    # 使用學習率排程器: 當 Loss 停滯時，自動降低學習率
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=500) 
    # 這裡簡化，暫時不用 Scheduler，先靠 Adam 自適應
    
    memory = ReplayMemory(MEMORY_SIZE)
    
    steps_done = 0
    recent_scores = deque(maxlen=50) # 只記錄最近 50 場的平均
    
    print(f"開始長程訓練... 目標: {NUM_EPISODES} 場 | 裝置: {device}")
    
    # 進度條
    progress_bar = tqdm(range(start_episode, NUM_EPISODES))
    
    for i_episode in progress_bar:
        env.reset()
        total_reward = 0
        
        while True:
            # --- 獲取狀態 ---
            next_states_dict = env.get_possible_next_states()
            
            if not next_states_dict:
                break # Game Over
            
            moves = list(next_states_dict.keys())
            states_np = list(next_states_dict.values())
            
            state_batch = torch.tensor(np.array(states_np), dtype=torch.float32).to(device)
            
            # --- Epsilon-Greedy 策略 ---
            # 使用指數衰減，讓 AI 在前 3000 場有較多探索機會
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            use_ai = True
            if random.random() < eps_threshold:
                use_ai = False
            
            # --- 決策 ---
            if use_ai:
                with torch.no_grad():
                    q_values = policy_net(state_batch)
                    best_idx = q_values.argmax().item()
            else:
                best_idx = random.randint(0, len(moves) - 1)
            
            action = moves[best_idx]
            chosen_state = states_np[best_idx]
            
            # --- 執行 ---
            reward, done = env.step(action)
            total_reward += reward
            
            # --- 記憶 ---
            if len(memory) > BATCH_SIZE:
                 transitions = memory.sample(BATCH_SIZE)
                 batch_states, _, _, batch_rewards, batch_dones = zip(*transitions)
                 
                 b_states = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
                 b_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device).unsqueeze(1)
                 
                 # 計算 Loss (Q-Learning Update)
                 current_q = policy_net(b_states)
                 target_q = b_rewards # 簡化版 Bellman: 假設這步的好壞就是 Reward
                 
                 loss = nn.MSELoss()(current_q, target_q)
                 
                 optimizer.zero_grad()
                 loss.backward()
                 
                 # 梯度裁剪 (避免梯度爆炸)
                 torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                 
                 optimizer.step()
            
            memory.push(chosen_state, action, None, reward, done)
            
            if done:
                break
        
        # --- 紀錄與顯示 ---
        recent_scores.append(total_reward)
        avg_score = sum(recent_scores) / len(recent_scores)
        
        progress_bar.set_description(f"Ep {i_episode} | R: {total_reward:.0f} | Avg: {avg_score:.0f} | Eps: {eps_threshold:.2f}")
        
        # --- 定期更新 Target Network ---
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # --- 定期存檔 (Checkpoint) ---
        if i_episode % CHECKPOINT_FREQ == 0:
            torch.save({
                'episode': i_episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "tetris_transformer_checkpoint.pth")
            # 同時存一個隨時可用的正式版
            torch.save(policy_net.state_dict(), "tetris_transformer.pth")

    # 訓練結束
    torch.save(policy_net.state_dict(), "tetris_transformer_final.pth")
    print("長程訓練完成！模型已儲存為 tetris_transformer_final.pth")

if __name__ == "__main__":
    train()
