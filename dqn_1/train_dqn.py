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

MAX_EPISODES = 80000        
EPS_START = 1.0        
EPS_END = 0.001          
EPS_DECAY_EPISODES = 60000 

MEMORY_SIZE = 30000        
BATCH_SIZE = 128
GAMMA = 0.95              
LR = 1e-4        

SAVE_PATH = 'tetris_dqn_new.pt' 

# ==========================================
# 2. 輔助函式
# ==========================================

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
# 3. 核心特徵提取
# ==========================================
def get_nuno_features(board, lines_cleared):
    rows, cols = config.rows, config.columns
    grid = np.array(board).reshape(rows, cols)
    
    # 1. Heights
    heights = []
    for c in range(cols):
        col_data = grid[:, c]
        if np.any(col_data == 2): 
            h = rows - np.argmax(col_data == 2)
            heights.append(h)
        else:
            heights.append(0)
            
    # 2. Wells
    wells = 0
    for c in range(cols):
        left_h = heights[c-1] if c > 0 else rows
        right_h = heights[c+1] if c < cols - 1 else rows
        my_h = heights[c]
        depth = min(left_h, right_h) - my_h
        if depth >= 2:
            wells += depth

    # 3. Holes
    holes_count = 0
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if grid[r][c] == 2:
                block_found = True
            elif block_found and grid[r][c] == 0:
                holes_count += 1
                
    # 4. Bumpiness
    bump_sum = 0
    for i in range(cols - 1):
        bump_sum += abs(heights[i] - heights[i+1])

    # 5. Normalization
    f_lines = float(lines_cleared) / 4.0
    f_holes = min(float(holes_count) / 20.0, 1.0)
    f_bumpiness = min(float(bump_sum) / 50.0, 1.0)
    f_height = min(float(sum(heights)) / 200.0, 1.0)
    f_wells = min(float(wells) / 20.0, 1.0)

    return np.array([f_lines, f_holes, f_bumpiness, f_height, f_wells], dtype=np.float32)

# ==========================================
# 4. 模型結構
# ==========================================
class NunoNet(nn.Module):
    def __init__(self):
        super().__init__()
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
    elite_memory = deque(maxlen=8000) 
    
    start_episode = 0
    best_score_so_far = 300.0 
    best_steps_so_far = 150

    if os.path.exists(SAVE_PATH):
        try:
            chk = torch.load(SAVE_PATH, weights_only=False)
            model.load_state_dict(chk['model'])
            optimizer.load_state_dict(chk['optimizer'])
            start_episode = chk['episode'] + 1
            if 'best_score' in chk:
                best_score_so_far = chk['best_score']
            if 'best_steps' in chk:
                best_steps_so_far = chk['best_steps']
            print(f"✅ Loaded checkpoint from Episode {start_episode}")
        except Exception as e:
            print(f"⚠️ Load failed ({e}), starting NEW training session.")

    print(f"--- Starting Elite Strategy Training (Total: {MAX_EPISODES}) ---")
    
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
        
        temp_game_memory = []
        
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
                
                sim_bump = sum(abs(get_column_height(s_sim.status, i) - get_column_height(s_sim.status, i+1)) for i in range(config.columns - 1))
                sim_max_h, sim_holes = get_raw_board_stats(s_sim.status)
                is_dangerous = (sim_holes > 2) or (sim_max_h >= 11)
                
                # 底薪設定 22.0
                r = 22.0 
                
                if is_dangerous:
                    if clears > 0:
                        r += clears * 20.0  
                    r -= sim_holes * 2.0   
                    r -= sim_max_h * 1.0    
                else:
                    if clears > 0:
                        if clears == 4:
                            r += 400.0 
                        else:
                            r += (clears ** 2) * config.columns
                    r -= sim_holes * 2.0
                    r -= sim_max_h * 0.5 
                    r -= sim_bump * 0.5   
                
                r -= sim_bump * 2.0

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
            
            if Handler.isDefeat(next_shot_obj, next_piece):
                reward = -1000.0
                game_over = True
                done = True
            else:
                done = False
                
            transition = (feat, reward, done, next_board_status)
            temp_game_memory.append(transition)
            
            shot = next_shot_obj
            piece = next_piece
            total_reward += reward
            step_count += 1
            
            if step_count > 5000: game_over = True 
            
            # 菁英混合訓練
            if len(memory) >= BATCH_SIZE and step_count % 5 == 0:
                if len(elite_memory) > BATCH_SIZE:
                    elite_count = int(BATCH_SIZE * 0.2)
                    normal_count = BATCH_SIZE - elite_count
                    batch_elite = random.sample(elite_memory, elite_count)
                    batch_normal = random.sample(memory, normal_count)
                    batch = batch_elite + batch_normal
                else:
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
        
        # === 遊戲結束結算 ===
        
        memory.extend(temp_game_memory)
        
        # 【修改】 菁英局判定邏輯：三者符合其一即可
        is_elite = False
        
        if total_reward > best_score_so_far:
            best_score_so_far = total_reward
            best_steps_so_far = step_count
            is_elite = True
            print(f"🌟 New High Score! {total_reward:.1f} ({step_count} steps) (Saved to Elite Memory)")
            
        elif total_reward > 1000.0:
            is_elite = True
            
        elif step_count > 200 and total_reward > -2000:
            # 【新增】 只要活過 400 步，哪怕分數不高，也值得存下來學習「怎麼不死」
            is_elite = True
            
        if is_elite:
            elite_memory.extend(temp_game_memory)

        if episode % 50 == 0:
            print(f"Ep {episode} | Score: {total_reward:.1f} | Best: {best_score_so_far:.1f} ({best_steps_so_far} steps) | Eps: {epsilon:.3f} | Steps: {step_count}")
            torch.save({
                'episode': episode,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_score_so_far,
                'best_steps': best_steps_so_far
            }, SAVE_PATH)

if __name__ == '__main__':
    train()