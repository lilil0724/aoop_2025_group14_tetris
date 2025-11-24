import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import config
import pieces
import shots
import Handler

# ==========================================
# 1. 參數設定
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Training on: {DEVICE}")

# Nuno 的圖表顯示約 1500 局後爆發，我們設定 3000 讓它有足夠時間
MAX_EPISODES = 6000        
EPS_START = 0.001          
EPS_END = 0.001           
EPS_DECAY_EPISODES = 100 

MEMORY_SIZE = 30000        
BATCH_SIZE = 512
GAMMA = 0.95              
LR = 1e-4               

SAVE_PATH = 'tetris_nuno.pt'

# ==========================================
# 2. 核心特徵提取 (加入關鍵的歸一化)
# ==========================================
def get_nuno_features(board, lines_cleared):
    """
    修正版特徵提取：
    將所有數值壓縮到 0.0 ~ 1.0 之間，避免神經網路被大數字(如高度)誤導。
    """
    rows, cols = config.rows, config.columns
    grid = np.array(board).reshape(rows, cols)
    
    # 計算每列高度
    heights = []
    for c in range(cols):
        col_data = grid[:, c]
        if np.any(col_data == 2): 
            h = rows - np.argmax(col_data == 2)
            heights.append(h)
        else:
            heights.append(0)
    
    # --- 特徵計算與歸一化 ---
    
    # 1. Lines Cleared (0 ~ 4) -> 除以 4.0
    f_lines = float(lines_cleared) / 4.0
    
    # 2. Holes (0 ~ 20+) -> 除以 20.0，超過算 1.0
    holes_count = 0
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if grid[r][c] == 2:
                block_found = True
            elif block_found and grid[r][c] == 0:
                holes_count += 1
    f_holes = min(float(holes_count) / 20.0, 1.0)
                
    # 3. Bumpiness (0 ~ 50+) -> 除以 50.0
    bump_sum = 0
    for i in range(cols - 1):
        bump_sum += abs(heights[i] - heights[i+1])
    f_bumpiness = min(float(bump_sum) / 50.0, 1.0)
        
    # 4. Total Height (0 ~ 200+) -> 除以 200.0
    # 這是最重要的一步，沒歸一化前，高度數值會輾壓其他特徵
    f_height = min(float(sum(heights)) / 200.0, 1.0)

    return np.array([f_lines, f_holes, f_bumpiness, f_height], dtype=np.float32)


# ==========================================
# 3. 模型結構
# ==========================================
class NunoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 4. 訓練主程式
# ==========================================
def train():
    model = NunoNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)
    
    start_episode = 0
    if os.path.exists(SAVE_PATH):
        try:
            chk = torch.load(SAVE_PATH)
            model.load_state_dict(chk['model'])
            optimizer.load_state_dict(chk['optimizer'])
            start_episode = chk['episode'] + 1
            print(f"✅ Loaded checkpoint from Episode {start_episode}")
            
            # === 【新增這兩行】 強制把讀檔回來的 LR 改成我們新設定的 1e-4 ===
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
            print(f"🔧 Learning Rate updated to {LR}")
            # ==========================================================
            
        except:
            print("⚠️ Checkpoint load failed, starting new.")

    print(f"--- Starting Nuno-Style Training (Total: {MAX_EPISODES}) ---")
    
    def get_epsilon(ep):
        if ep > EPS_DECAY_EPISODES: return EPS_END
        decrease = (EPS_START - EPS_END) / EPS_DECAY_EPISODES
        return EPS_START - (decrease * ep)

    for episode in range(start_episode, MAX_EPISODES):
        shot = shots.Shot()
        piece = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
        
        epsilon = get_epsilon(episode)
        
        step_count = 0
        total_reward = 0
        game_over = False
        
        while not game_over:
            legal_moves = []
            rots = len(config.shapes[piece.shape])
            if piece.shape == 'O': rots = 1
            elif piece.shape in ['S', 'Z', 'I']: rots = 2
            
            for rot in range(rots):
                t_piece = pieces.Piece(piece.x, piece.y, piece.shape)
                t_piece.rotation = rot
                for x in range(-2, config.columns + 1):
                    t_piece.x = x
                    if Handler.isValidPosition(shot, t_piece):
                        legal_moves.append((x, rot))
            
            if not legal_moves:
                game_over = True
                break
                
            candidates = [] 
            for action in legal_moves:
                s_sim = copy.deepcopy(shot)
                p_sim = copy.deepcopy(piece)
                p_sim.x, p_sim.rotation = action
                Handler.instantDrop(s_sim, p_sim)
                clears, _ = Handler.eliminateFilledRows(s_sim, p_sim)
                
                # === Nuno 原版獎勵 (指數成長) ===
                # 我們保留這個指數獎勵，因為這是產生 "爆發性高分" 的關鍵
                r = 1.0 + (clears ** 2) * config.columns
                
                f = get_nuno_features(s_sim.status, clears)
                candidates.append((f, r, s_sim.status, action))

            if random.random() < epsilon:
                chosen = random.choice(candidates)
            else:
                model.eval()
                with torch.no_grad():
                    feats = [c[0] for c in candidates]
                    b_f = torch.tensor(np.array(feats), dtype=torch.float32, device=DEVICE)
                    q_vals = model(b_f).squeeze(-1)
                    best_idx = torch.argmax(q_vals).item()
                chosen = candidates[best_idx]
                
            feat, reward, next_board_status, action = chosen
            
            next_piece = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
            next_shot_obj = shots.Shot()
            next_shot_obj.status = next_board_status 
            
            # === 【關鍵修正】 死亡懲罰加重 ===
            if Handler.isDefeat(next_shot_obj, next_piece):
                # 之前是 -2.0，改成 -100.0
                # 這樣它就算消了一次 Tetris (160分)，也不會覺得死掉是划算的
                reward = -100.0 
                game_over = True
                done = True
            else:
                done = False
                
            memory.append((feat, reward, done, next_board_status))
            
            shot = next_shot_obj
            piece = next_piece
            total_reward += reward
            step_count += 1
            
            if step_count > 5000: game_over = True 
            
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_f, b_r, b_d, b_next_st = zip(*batch)
                
                t_f = torch.tensor(np.array(b_f), dtype=torch.float32, device=DEVICE)
                t_r = torch.tensor(b_r, dtype=torch.float32, device=DEVICE)
                t_d = torch.tensor(b_d, dtype=torch.float32, device=DEVICE)
                
                model.eval()
                with torch.no_grad():
                    next_feats = []
                    for st in b_next_st:
                        next_feats.append(get_nuno_features(st, 0))
                    t_next_f = torch.tensor(np.array(next_feats), dtype=torch.float32, device=DEVICE)
                    q_next = model(t_next_f).squeeze(-1)
                
                target = t_r + GAMMA * q_next * (1 - t_d)
                
                model.train()
                q_pred = model(t_f).squeeze(-1)
                loss = criterion(q_pred, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 50 == 0:
            print(f"Ep {episode} | Score: {total_reward:.1f} | Eps: {epsilon:.3f} | Steps: {step_count}")
            torch.save({
                'episode': episode,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, SAVE_PATH)

if __name__ == '__main__':
    train()