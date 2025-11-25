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

MAX_EPISODES = 30000        
EPS_START = 0.3         
EPS_END = 0.001          
EPS_DECAY_EPISODES = 26000

# 修改：加大記憶體，縮小 Batch Size 以穩定初期學習
MEMORY_SIZE = 30000        
BATCH_SIZE = 128
GAMMA = 0.95              
LR = 1e-4              # 重新訓練建議先用較大的 LR (1e-3)，之後微調可改 1e-4

SAVE_PATH = 'tetris_dqn_new.pt' # 建議改名，避免讀到舊的格式報錯

# ==========================================
# 2. 輔助函式
# ==========================================

# 新增：計算單列高度的小工具 (給獎勵函數用)
def get_column_height(board, col_idx):
    rows = config.rows
    for r in range(rows):
        if board[r][col_idx] == 2:
            return rows - r
    return 0

def get_raw_board_stats(board):
    rows, cols = config.rows, config.columns
    grid = np.array(board).reshape(rows, cols)
    
    heights = []
    for c in range(cols):
        col_data = grid[:, c]
        if np.any(col_data == 2): 
            h = rows - np.argmax(col_data == 2)
            heights.append(h)
        else:
            heights.append(0)
            
    max_height = max(heights)
    
    holes = 0
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if grid[r][c] == 2:
                block_found = True
            elif block_found and grid[r][c] == 0:
                holes += 1
                
    return max_height, holes

# ==========================================
# 3. 核心特徵提取 (修正版：5 特徵)
# ==========================================
def get_nuno_features(board, lines_cleared):
    rows, cols = config.rows, config.columns
    grid = np.array(board).reshape(rows, cols)
    
    # --- 1. 計算每行高度 (Heights) ---
    heights = []
    for c in range(cols):
        col_data = grid[:, c]
        if np.any(col_data == 2): 
            h = rows - np.argmax(col_data == 2)
            heights.append(h)
        else:
            heights.append(0)
            
    # --- 2. 計算深井 (Wells) [原本缺少的!] ---
    wells = 0
    for c in range(cols):
        left_h = heights[c-1] if c > 0 else rows
        right_h = heights[c+1] if c < cols - 1 else rows
        my_h = heights[c]
        depth = min(left_h, right_h) - my_h
        if depth >= 2:
            wells += depth

    # --- 3. 計算坑洞 (Holes) ---
    holes_count = 0
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if grid[r][c] == 2:
                block_found = True
            elif block_found and grid[r][c] == 0:
                holes_count += 1
                
    # --- 4. 計算表面凹凸 (Bumpiness) ---
    bump_sum = 0
    for i in range(cols - 1):
        bump_sum += abs(heights[i] - heights[i+1])

    # --- 5. 歸一化 (Normalization) ---
    f_lines = float(lines_cleared) / 4.0
    f_holes = min(float(holes_count) / 20.0, 1.0)
    f_bumpiness = min(float(bump_sum) / 50.0, 1.0)
    f_height = min(float(sum(heights)) / 200.0, 1.0)
    f_wells = min(float(wells) / 20.0, 1.0) # 新增這個

    # 回傳完整的 5 個特徵
    return np.array([f_lines, f_holes, f_bumpiness, f_height, f_wells], dtype=np.float32)

# ==========================================
# 4. 模型結構 (修正版：輸入 5)
# ==========================================
class NunoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改：Input 5 features, Hidden layer 64
        self.net = nn.Sequential(
            nn.Linear(5, 64),  
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        # 技巧：讓最後一層 Bias 稍微偏正，鼓勵活著
        with torch.no_grad():
             self.net[-1].bias.fill_(0.1)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 5. 訓練主程式
# ==========================================
def train():
    model = NunoNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)
    
    start_episode = 0
    # 嘗試載入舊檔，如果形狀不對(報錯)就從頭開始
    if os.path.exists(SAVE_PATH):
        try:
            chk = torch.load(SAVE_PATH)
            model.load_state_dict(chk['model'])
            optimizer.load_state_dict(chk['optimizer'])
            start_episode = chk['episode'] + 1
            print(f"✅ Loaded checkpoint from Episode {start_episode}")
        except Exception as e:
            print(f"⚠️ Load failed ({e}), starting NEW training session.")

    print(f"--- Starting Adaptive Strategy Training (Total: {MAX_EPISODES}) ---")
    
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
                
                # === [修改重點] 動態策略與獎勵計算 ===
                
                # 1. 取得盤面原始狀態 (用於判斷是否危險)
                sim_max_h, sim_holes = get_raw_board_stats(s_sim.status)
                
                # 2. 危險判定
                is_dangerous = (sim_holes > 2) or (sim_max_h >= 7)
                
                r = 15.0 # 只要活著就有 1 分
                
                if is_dangerous:
                    # === [策略 A: 保守/恐慌模式] ===
                    if clears > 0:
                        r += clears * 20.0  # 有消就好
                    
                    # 懲罰：在危險模式下，對「洞」和「高度」重罰
                    r -= sim_holes * 1.5   
                    r -= sim_max_h * 0.8    
                    
                else:
                    # === [策略 B: 激進/貪婪模式] ===
                    if clears > 0:
                        # 指數級獎勵：消4行(Tetris) 給極高分
                        if clears == 4:
                            r += 300.0
                        else:
                            r += (clears ** 2) * config.columns
                    
                    # 安全模式下，稍微容忍深井 (為了 Tetris)
                    r -= sim_holes * 0.5 
                    r -= sim_max_h * 0.3    
                
                # 額外：加上 Bumpiness (表面凹凸) 懲罰
                sim_bump = sum(abs(get_column_height(s_sim.status, i) - get_column_height(s_sim.status, i+1)) for i in range(config.columns - 1))
                r -= sim_bump * 0.5

                # 3. 獲取特徵 (這是修正後的 5 特徵版本!)
                f = get_nuno_features(s_sim.status, clears)
                
                candidates.append((f, r, s_sim.status, action))

            # Epsilon-Greedy 選擇
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
            
            # === [修改重點] 死亡懲罰加重 ===
            if Handler.isDefeat(next_shot_obj, next_piece):
                reward = -1000.0   # 讓 AI 極度恐懼死亡
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
            
            # 訓練步 (Experience Replay)
            if len(memory) >= BATCH_SIZE and step_count % 5 == 0: # 加快訓練頻率
                batch = random.sample(memory, BATCH_SIZE)
                b_f, b_r, b_d, b_next_st = zip(*batch)
                
                t_f = torch.tensor(np.array(b_f), dtype=torch.float32, device=DEVICE)
                t_r = torch.tensor(b_r, dtype=torch.float32, device=DEVICE)
                t_d = torch.tensor(b_d, dtype=torch.float32, device=DEVICE)
                
                model.eval()
                with torch.no_grad():
                    next_feats = []
                    for st in b_next_st:
                        next_feats.append(get_nuno_features(st, 0)) # Next state 預設 clears=0
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