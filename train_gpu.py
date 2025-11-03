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

import config
import Handler
import pieces
import shots
# (注意) 確保你使用的是 JIT 加速的 'np' 版本
from ai_player_nn import ValueNet, _extract_features_from_board_np 

# -----------------------
# 參數設定 (V4 方案)
# -----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.985
LR = 1e-4                # (V4 修正) 1e-4 更穩定
BUFFER_SIZE = 50000      # (保留) 您的 50k
BATCH_SIZE = 512         # (保留) 您的 512
TRAIN_STEPS = 150000     # (V4 修正) 增加總步數
TARGET_UPDATE = 5000     # (V4 修正) 5k 步更穩定
REWARD_SCALE = 40.0      # (*** 關鍵修正 ***) 獎勵基準是 40 (1 line)

EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 120000       # (*** 關鍵修正 ***) 8 萬步太短，給 40 萬步探索
MAX_MOVES_PER_GAME = 1500 # (V4 修正) 讓 AI 能堆更高

SAVE_PATH = 'tetris_valuenet_v4.pt' # (V4 修正) 存成新檔名
CHECKPOINT_PATH = 'tetris_checkpoints_v4/' # (V4 修正) 存到新資料夾
CHECKPOINT_INTERVAL = 30000
GRAD_ACCUM_STEPS = 4      # (保留) 您的梯度累加
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
        # (修正) 確保 tmpl 是從 piece 複製
        tmpl = pieces.Piece(piece.x, piece.y, piece.shape) 
        tmpl.rotation = rot
        for x in range(-2, config.columns + 1):
            tmpl.x = x
            if Handler.isValidPosition(shot, tmpl):
                moves.append((x, rot))
    return moves


# (*** 關鍵修正 ***)
# 刪除舊的 SCORES = {...} 字典

def _simulate_step(shot, piece, action):
    """
    V4 修正版：
    1. 使用 config.py 的 40/100/300/1200 分作為基礎獎勵
    2. 使用我們之前調校好的「溫和懲罰」
    """
    prev_score = shot.score
    
    # (*** 速度優化 ***) 
    # 確保使用 .copy() 而不是 deepcopy
    if hasattr(shot, 'copy'):
        sim_shot = shot.copy()
    else:
        sim_shot = copy.deepcopy(shot) # (備用)
        
    sim_piece = copy.deepcopy(piece)

    sim_piece.x, sim_piece.rotation = action
    Handler.instantDrop(sim_shot, sim_piece)
    
    # (*** 關鍵 ***) 
    # 這裡會使用 config.py 的 {1:40, 4:1200} 來更新 sim_shot.score
    clears, all_clear = Handler.eliminateFilledRows(sim_shot, sim_piece) 

    next_shape = random.choice(list(config.shapes.keys()))
    next_piece = pieces.Piece(5, 0, next_shape)
    done = Handler.isDefeat(sim_shot, next_piece)

    # 1. 基礎獎勵 (來自 config.py: 40, 100, 300, 1200)
    reward = sim_shot.score - prev_score 

    # 2. 溫和的懲罰 (v3 版的參數)
    if not done:
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

    # (移除舊的 reward += 0.02)
    return sim_shot, reward, done, next_piece


# -----------------------
# 主訓練流程
# -----------------------
def train():
    in_dim = config.rows * config.columns + config.columns + (config.columns - 1) + 6
    policy = ValueNet(in_dim).to(DEVICE)
    if USE_COMPILE:
        policy = torch.compile(policy)
    target = ValueNet(in_dim).to(DEVICE)
    if USE_COMPILE:
        target = torch.compile(target)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimiz = optim.Adam(policy.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    scaler = GradScaler()

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    replay = deque(maxlen=BUFFER_SIZE)

    step = 0
    episode = 0
    running_losses = deque(maxlen=LOG_INTERVAL)
    accum_counter = 0
    episode_scores = deque(maxlen=EPISODE_LOG_WINDOW)

    print(f"--- V4 方案開始訓練 (共 {TRAIN_STEPS} 步) ---")
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
                simulation_results, features_list, rewards_list = [], [], []
                for a in legal_moves:
                    sim_shot, r, d, nxt_p = _simulate_step(shot, piece, a)
                    simulation_results.append((a, sim_shot, r, d, nxt_p))
                    features_list.append(_extract_features_from_board_np(sim_shot.status))
                    rewards_list.append(r)

                if not features_list: # (安全檢查)
                    break 

                with torch.no_grad(), autocast():
                    state_batch = torch.tensor(features_list, dtype=torch.float32, device=DEVICE)
                    values_next = policy(state_batch).squeeze()
                    scaled_r = torch.tensor([r / REWARD_SCALE for r in rewards_list],
                                            dtype=torch.float32, device=DEVICE)
                    q_values = scaled_r + GAMMA * values_next
                    best_idx = torch.argmax(q_values).item()

                action, next_shot, reward, done, new_next_piece = simulation_results[best_idx]

            current_episode_score += reward
            # (*** 關鍵 ***) 
            # original_episode_score 現在會記錄 40, 100 等分數
            original_episode_score += (next_shot.score - shot.score) 

            x_feats = torch.tensor(_extract_features_from_board_np(shot.status), dtype=torch.float32)
            next_x_feats = torch.tensor(_extract_features_from_board_np(next_shot.status), dtype=torch.float32)
            scaled_reward = torch.tensor([reward / REWARD_SCALE], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.float32)
            replay.append((x_feats, scaled_reward, done_tensor, next_x_feats))

            shot, piece = next_shot, new_next_piece
            moves += 1
            step += 1

            # 訓練批次
            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)
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
            
            # (*** 修正日誌 ***)
            # 確保 'original_episode_score' 被正確記錄
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
                    ckpt_name = f"tetris_valuenet_step_{step}.pt"
                    ckpt_save_path = os.path.join(CHECKPOINT_PATH, ckpt_name)
                    torch.save(policy.state_dict(), ckpt_save_path)
                    print(f"--- [Step {step:7d}] Checkpoint saved to {ckpt_save_path} ---")

            if done:
                break

        episode += 1
        # (*** 關鍵 ***)
        # 我們記錄的是 original_episode_score，這才是真正的遊戲分數
        episode_scores.append(original_episode_score) 
        shot, piece, next_piece, current_episode_score = _env_reset()
        original_episode_score = 0

    torch.save(policy.state_dict(), SAVE_PATH)
    print(f"✅ 訓練完成！已儲存模型至: {SAVE_PATH}")


if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    train()