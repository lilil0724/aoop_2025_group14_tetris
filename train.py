import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

from tetris_env import TetrisEnv
from ai_model import TetrisActorCritic # 記得改 import
import config

# --- A2C 參數 ---
LR = 1e-4
GAMMA = 0.99
NUM_EPISODES = 10000
ENTROPY_BETA = 0.01 # 鼓勵探索 (避免演員太早只會出一招)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_a2c():
    env = TetrisEnv()
    model = TetrisActorCritic(config.rows, config.columns).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"開始 A2C 訓練... Device: {device}")
    progress_bar = tqdm(range(NUM_EPISODES))
    
    for i_episode in progress_bar:
        env.reset()
        
        # 儲存這一場的歷程
        log_probs = []    # 演員的信心指數
        values = []       # 評論家的評分
        rewards = []      # 實際拿到的獎勵
        
        total_reward = 0
        
        while True:
            # 1. 獲取所有可能的下一步
            next_states_dict = env.get_possible_next_states()
            if not next_states_dict: break
            
            moves = list(next_states_dict.keys())
            states_np = list(next_states_dict.values())
            
            # 轉 Tensor
            state_batch = torch.tensor(np.array(states_np), dtype=torch.float32).to(device)
            
            # 2. 模型思考 (Forward Pass)
            # 這次我們得到兩個輸出: policy_logits (喜好度) 和 value (預期分數)
            logits, value_est = model(state_batch)
            
            # --- 關鍵差異: Softmax 選擇動作 ---
            # 演員給每個可能的盤面打「喜好分」，我們用 Softmax 轉成機率
            probs = F.softmax(logits.view(-1), dim=0)
            
            # 根據機率抽樣 (不再是 Epsilon-Greedy 硬幣了！)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            
            # 紀錄這一刻的想法
            log_prob = dist.log_prob(action_idx)
            selected_value = value_est[action_idx] # 評論家對「選中這步」的評價
            
            log_probs.append(log_prob)
            values.append(selected_value)
            
            # 3. 執行動作
            action = moves[action_idx.item()]
            reward, done = env.step(action)
            rewards.append(reward)
            total_reward += reward
            
            if done: break
            
        # --- 這一場結束後，開始總檢討 (Backpropagation) ---
        
        # 1. 計算回報 (Returns) - 從最後一步往前推
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # 整理資料
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).view(-1)
        
        # 標準化 Returns (讓訓練更穩定)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 2. 計算優勢 (Advantage)
        # Advantage = 實際結果 - 評論家的預測
        # 如果實際結果比評論家想得好，Advantage > 0，我們要鼓勵演員多這樣做
        advantage = returns - values.detach()
        
        # 3. 計算 Loss
        # Actor Loss: -log_prob * advantage (鼓勵 Advantage 高的動作)
        actor_loss = -(log_probs * advantage).mean()
        
        # Critic Loss: 預測值要越接近實際值越好
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy Loss: 鼓勵機率分佈分散一點，不要太早太武斷 (探索機制)
        # (這裡沒實作完整 Entropy 計算以簡化，但概念是這樣)
        
        total_loss = actor_loss + 0.5 * critic_loss
        
        # 4. 更新模型
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 顯示進度
        progress_bar.set_description(f"Ep {i_episode} | R: {total_reward:.0f}")
        
        if i_episode % 1000 == 0:
            torch.save(model.state_dict(), "tetris_a2c.pth")

    torch.save(model.state_dict(), "tetris_a2c.pth")
    print("A2C 訓練完成！")

if __name__ == "__main__":
    train_a2c()
