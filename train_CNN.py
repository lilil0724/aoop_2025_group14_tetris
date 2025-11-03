import os
import copy
import random
import json
from collections import deque
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np # <-- (新增) 為了預處理
import glob
import re

import config
import Handler
import pieces
import shots
# (注意) 舊的特徵擷取 *僅* 用於「獎勵計算」，*不* 用於 CNN 輸入
from ai_player_nn import _extract_features_from_board_np 

# -----------------------
# (新增) CNN 模型定義
# -----------------------
class CNNValueNet(nn.Module):
    """
    直接接收 (B, 1, 20, 10) 盤面狀態的 CNN 值網路
    """
    def __init__(self, in_channels=1, h=config.rows, w=config.columns):
        super().__init__()
        
        # 卷積層：用來 "看" 盤面
        self.conv_stack = nn.Sequential(
            # Input: (B, 1, 20, 10)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            # -> (B, 32, 20, 10)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # -> (B, 64, 10, 5)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # -> (B, 128, 5, 2)
        )
        
        # 根據最終的 H, W 計算扁平化後的大小
        # H: 20 -> (pool 2) 10 -> (pool 2) 5
        # W: 10 -> (pool 2) 5  -> (pool 2) 2 (整數除法)
        self.flat_size = 128 * 5 * 2 # 1280

        # 全連接層：用來 "決策"
        self.fc_stack = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 輸出單一的 "盤面價值"
        )

    def forward(self, x):
        # x 的形狀: (B, 1, 20, 10)
        x = self.conv_stack(x)
        x = torch.flatten(x, 1) # 扁平化 (B, 1280)
        x = self.fc_stack(x)
        return x

# -----------------------
# (新增) CNN 預處理函式
# -----------------------
def _preprocess_board_for_cnn(board_status):
    """
    將 2D list 盤面 (0=empty, 2=fixed) 轉換為 (1, H, W) 的 0/1 張量
    """
    # 轉換為 NumPy array (20, 10)
    board_np = np.array(board_status, dtype=np.uint8)
    
    # 建立 (20, 10) 的張量, 0=empty, 1=fixed
    # (status == 2) -> 1.0, else 0.0
    board_tensor = (board_np == 2).astype(np.float32)
    
    # 增加 channel 維度 -> (1, 20, 10)
    return np.expand_dims(board_tensor, axis=0)

# -----------------------
# 參數設定 (V4 方案)
# -----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.985
LR = 1e-4                
BUFFER_SIZE = 50000      
BATCH_SIZE = 512         
TRAIN_STEPS = 300000     
TARGET_UPDATE = 5000     
REWARD_SCALE = 40.0      

EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 260000       
MAX_MOVES_PER_GAME = 1500 

SAVE_PATH = 'tetris_valuenet_cnn_v1.pt' # (修改) 存成 CNN 檔名
CHECKPOINT_PATH = 'tetris_checkpoints_cnn_v1/' # (修改) 存到 CNN 資料夾
CHECKPOINT_INTERVAL = 50000
GRAD_ACCUM_STEPS = 4      
USE_COMPILE = False
LOG_INTERVAL = 1000
EPISODE_LOG_WINDOW = 50

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def epsilon_by_step(step):
    if step >= EPS_DECAY:
        return EPS_END
    return EPS_START + (EPS_END - EPS_START) * (step / EPS_DECAY)


def _env_reset():
    s = shots.Shot()
    shape1 = random.choice(list(config.shapes.keys()))
    cur = pieces.Piece(5, 0, shape1)
    shape2 = random.choice(list(config.shapes.keys()))
    nxt = pieces.Piece(5, 0, shape2)
    return s, cur, nxt, 0


def _enumerate_legal_moves(shot, piece):
    moves = []
    for rot in range(len(config.shapes[piece.shape])):
        tmpl = pieces.Piece(piece.x, piece.y, piece.shape) 
        tmpl.rotation = rot
        for x in range(-2, config.columns + 1):
            tmpl.x = x
            if Handler.isValidPosition(shot, tmpl):
                moves.append((x, rot))
    return moves


def _simulate_step(shot, piece, action):
    """
    V4 修正版 (與您原版相同)：
    1. 使用 config.py 的 40/100/300/1200 分作為基礎獎勵
    2. 使用我們之前調校好的「溫和懲罰」
    (註: 這裡 *仍然* 需要 _extract_features_from_board_np 來計算懲罰)
    """
    prev_score = shot.score
    
    if hasattr(shot, 'copy'):
        sim_shot = shot.copy()
    else:
        sim_shot = copy.deepcopy(shot) 
        
    sim_piece = copy.deepcopy(piece)

    sim_piece.x, sim_piece.rotation = action
    Handler.instantDrop(sim_shot, sim_piece)
    
    clears, all_clear = Handler.eliminateFilledRows(sim_shot, sim_piece) 

    next_shape = random.choice(list(config.shapes.keys()))
    next_piece = pieces.Piece(5, 0, next_shape)
    done = Handler.isDefeat(sim_shot, next_piece)

    # 1. 基礎獎勵 (來自 config.py: 40, 100, 300, 1200)
    reward = sim_shot.score - prev_score 

    # 2. 溫和的懲罰 (v3 版的參數)
    if not done:
        # *** (注意) ***
        # 這裡仍然使用舊的特徵擷取來 "計算獎勵"
        feats = _extract_features_from_board_np(sim_shot.status) 
        flat_grid_size = config.rows * config.columns
        heights_size = config.columns
        deltas_size = config.columns - 1
        
        holes = feats[flat_grid_size]
        bumpiness = feats[flat_grid_size + heights_size + deltas_size + 2] 

        reward -= holes * 0.5       # (溫和懲罰)
        reward -= bumpiness * 0.1   # (溫和懲罰)

    # 3. All Clear 獎勵
    if all_clear:
        reward += 50.0 # (PC 獎勵)

    return sim_shot, reward, done, next_piece


# -----------------------
# 主訓練流程
# -----------------------
def train():
    # (修改) 移除 in_dim，改用 CNN
    step = 0
    latest_checkpoint = None
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    # 2. 尋找所有 .pt 檔案
    #    (使用您截圖中的路徑)
    files = glob.glob(os.path.join(CHECKPOINT_PATH, "tetris_cnn_step_*.pt"))
    
    if files:
        # 3. 根據檔名中的 "step 數字" 排序，找到最新的
        try:
            files.sort(key=lambda f: int(re.search(r'step_(\d+)\.pt', f).group(1)), reverse=True)
            latest_checkpoint = files[0]
            
            # 4. (關鍵) 從檔名恢復 step 數字
            step = int(re.search(r'step_(\d+)\.pt', latest_checkpoint).group(1))
            print(f"✅ 找到 Checkpoint，將從 [Step {step}] 繼續訓練...")
        except Exception as e:
            print(f"⚠️ 找到 Checkpoint，但無法解析 step 數: {e}. 將從 step 0 開始。")
            step = 0 # 解析失敗則重頭
            
    policy = CNNValueNet().to(DEVICE)
    if USE_COMPILE:
        policy = torch.compile(policy)
    target = CNNValueNet().to(DEVICE)
    if USE_COMPILE:
        target = torch.compile(target)
        
    if latest_checkpoint:
        try:
            policy.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE))
            print(f"✅ 成功載入模型權重: {latest_checkpoint}")
        except Exception as e:
            print(f"⚠️ 載入模型失敗: {e}. 將使用隨機初始化的新模型。")
            step = 0 # 載入失敗，step 歸零
    else:
        print("ℹ️ 未找到任何 Checkpoint，將從頭開始訓練新模型。")
        
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimiz = optim.Adam(policy.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    scaler = GradScaler()

    #os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    replay = deque(maxlen=BUFFER_SIZE)

    #step = 0
    episode = 0
    running_losses = deque(maxlen=LOG_INTERVAL)
    accum_counter = 0
    episode_scores = deque(maxlen=EPISODE_LOG_WINDOW)

    print(f"--- (CNN V1) 方案開始訓練 (共 {TRAIN_STEPS} 步) ---")
    if step > 0:
        print(f"--- (將從 {step} 步繼續) ---")
    print(f"Device={DEVICE}, LR={LR}, RewardScale={REWARD_SCALE}, EpsDecay={EPS_DECAY}")

    shot, piece, next_piece, current_episode_score = _env_reset()
    original_episode_score = 0

    while step < TRAIN_STEPS:
        moves = 0
        while moves < MAX_MOVES_PER_GAME and step < TRAIN_STEPS:
            legal_moves = _enumerate_legal_moves(shot, piece)
            if not legal_moves:
                break

            eps = epsilon_by_step(step)
            if random.random() < eps:
                action = random.choice(legal_moves)
                next_shot, reward, done, new_next_piece = _simulate_step(shot, piece, action)
            else:
                # (修改) 儲存 CNN 輸入 和 獎勵
                simulation_results, cnn_inputs_list, rewards_list = [], [], []
                
                for a in legal_moves:
                    sim_shot, r, d, nxt_p = _simulate_step(shot, piece, a)
                    simulation_results.append((a, sim_shot, r, d, nxt_p))
                    # (修改) 使用新的 CNN 預處理
                    cnn_inputs_list.append(_preprocess_board_for_cnn(sim_shot.status))
                    rewards_list.append(r)

                if not cnn_inputs_list: # (安全檢查)
                    break 

                with torch.no_grad(), autocast():
                    # (修改) state_batch 現在是 (N_moves, 1, 20, 10)
                    state_batch = torch.tensor(np.array(cnn_inputs_list), dtype=torch.float32, device=DEVICE)
                    
                    values_next = policy(state_batch).squeeze()
                    scaled_r = torch.tensor([r / REWARD_SCALE for r in rewards_list],
                                            dtype=torch.float32, device=DEVICE)
                    q_values = scaled_r + GAMMA * values_next
                    best_idx = torch.argmax(q_values).item()

                action, next_shot, reward, done, new_next_piece = simulation_results[best_idx]

            current_episode_score += reward
            original_episode_score += (next_shot.score - shot.score) 

            # (修改) 儲存 CNN 格式的 (S, A, R, S')
            x_feats = torch.tensor(_preprocess_board_for_cnn(shot.status), dtype=torch.float32)
            next_x_feats = torch.tensor(_preprocess_board_for_cnn(next_shot.status), dtype=torch.float32)
            
            scaled_reward = torch.tensor([reward / REWARD_SCALE], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.float32)
            replay.append((x_feats, scaled_reward, done_tensor, next_x_feats))

            shot, piece = next_shot, new_next_piece
            moves += 1
            step += 1

            # 訓練批次
            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)
                # b_s 和 b_s_next 現在會是 (BatchSize, 1, 20, 10)
                b_s, b_r, b_d, b_s_next = map(lambda x: torch.stack(x).to(DEVICE, non_blocking=True), zip(*batch))
                
                with torch.no_grad():
                    v_next = target(b_s_next).squeeze()
                    y_target = b_r.squeeze() + GAMMA * v_next * (1 - b_d.squeeze())

                with autocast():
                    pred = policy(b_s).squeeze()
                    loss = loss_fn(pred, y_target)
                running_losses.append(loss.item())

                loss = loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                accum_counter += 1
                if accum_counter % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimiz)
                    nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                    scaler.step(optimiz)
                    scaler.update()
                    optimiz.zero_grad(set_to_none=True)

            if step % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())
                print(f"--- [Step {step:7d}] Target network updated. ---")
            
            if step % LOG_INTERVAL == 0:
                avg_loss = statistics.mean(running_losses) if running_losses else 0
                avg_score = statistics.mean(episode_scores) if episode_scores else 0
                
                print(f"[Step {step:7d}/{TRAIN_STEPS}] "
                      f"AvgLoss={avg_loss:8.6f} | AvgScore={avg_score:8.2f} | "
                      f"Eps={eps:.4f} | Buffer={len(replay)}")
                
                if len(episode_scores) >= 10:
                    max_score = max(episode_scores)
                    print(f"[Monitor] Recent max score: {max_score:.2f}")

                if step > 0 and step % CHECKPOINT_INTERVAL == 0:
                    ckpt_name = f"tetris_cnn_step_{step}.pt" # (修改) 檔名
                    ckpt_save_path = os.path.join(CHECKPOINT_PATH, ckpt_name)
                    torch.save(policy.state_dict(), ckpt_save_path)
                    print(f"--- [Step {step:7d}] Checkpoint saved to {ckpt_save_path} ---")

            if done:
                break

        episode += 1
        episode_scores.append(original_episode_score) 
        shot, piece, next_piece, current_episode_score = _env_reset()
        original_episode_score = 0

    torch.save(policy.state_dict(), SAVE_PATH)
    print(f"✅ 訓練完成！已儲存模型至: {SAVE_PATH}")


if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    train()