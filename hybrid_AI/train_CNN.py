import os
import copy
import random
from collections import deque
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import glob
import re

import config
import Handler
import pieces
import shots
# 引入我們剛更新的 ai_player_nn
from ai_player_nn import HybridValueNet, get_hybrid_input, _extract_features_from_board_np

# -----------------------
# 參數設定 (Hybrid V1)
# -----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.99             # 稍微提高遠見
LR = 1e-4
BUFFER_SIZE = 60000      # 加大 Buffer
BATCH_SIZE = 512
TRAIN_STEPS = 500000     # 增加總步數
TARGET_UPDATE = 2000
REWARD_SCALE = 30.0      # 調整 Scaling

EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 250000

MAX_MOVES_PER_GAME = 4000

# 存檔路徑 (建議改名以免覆蓋舊版)
SAVE_PATH = 'tetris_hybrid_v1.pt'
CHECKPOINT_PATH = 'tetris_checkpoints_hybrid_v1/'
CHECKPOINT_INTERVAL = 50000
GRAD_ACCUM_STEPS = 1     # 簡化累積
LOG_INTERVAL = 1000

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

def _get_board_stats(board):
    """ 取得計算獎勵需要的原始特徵值 """
    feats = _extract_features_from_board_np(board)
    # feats 順序參考 ai_player_nn: 
    # holes(-6), agg_height(-5), bumpiness(-4), max_height(-3)
    return {
        'holes': feats[-6],
        'agg_height': feats[-5],
        'bumpiness': feats[-4],
        'max_height': feats[-3]
    }

def _simulate_step(shot, piece, action):
    # 1. 複製環境
    if hasattr(shot, 'copy'):
        sim_shot = shot.copy()
    else:
        sim_shot = copy.deepcopy(shot) 
    sim_piece = copy.deepcopy(piece)
    
    # 2. 【關鍵】記錄移動前的盤面狀態 (用於計算差值)
    prev_stats = _get_board_stats(sim_shot.status)

    # 3. 執行動作
    sim_piece.x, sim_piece.rotation = action
    Handler.instantDrop(sim_shot, sim_piece)
    clears, all_clear = Handler.eliminateFilledRows(sim_shot, sim_piece) 

    next_shape = random.choice(list(config.shapes.keys()))
    next_piece = pieces.Piece(5, 0, next_shape)
    done = Handler.isDefeat(sim_shot, next_piece)

    # 4. 【關鍵】記錄移動後的盤面狀態
    curr_stats = _get_board_stats(sim_shot.status)
    
    # 5. 計算新版獎勵
    reward = 0.0
    
    # --- A. 消行獎勵 (鼓勵多行消除) ---
    if clears == 1: reward += 10.0
    elif clears == 2: reward += 30.0
    elif clears == 3: reward += 60.0
    elif clears == 4: reward += 100.0
    
    if all_clear: reward += 200.0

    # --- B. 差分獎勵 (Potential-Based Reward Shaping) ---
    # 核心概念：獎勵「改善」，懲罰「惡化」，而不是懲罰「現狀」。
    
    # 洞數減少是好事 (+)，增加是壞事 (-)
    reward += (prev_stats['holes'] - curr_stats['holes']) * 8.0       
    
    # 平整度變好是好事 (+)
    reward += (prev_stats['bumpiness'] - curr_stats['bumpiness']) * 2.0 
    
    # 高度降低是好事 (+)
    reward += (prev_stats['agg_height'] - curr_stats['agg_height']) * 1.5 
    
    # --- C. 輕微的狀態壓力 ---
    # 這是唯一保留的「絕對值懲罰」，但權重很輕。
    # 目的是防止 AI 在高空無限空轉而不消除。
    reward -= curr_stats['max_height'] * 0.1

    # --- D. 死亡懲罰 ---
    # 降低懲罰力度，避免 AI 為了逃避長期扣分而選擇自殺
    if done:
        reward -= 50.0 

    return sim_shot, reward, done, next_piece

def train():
    step = 0
    latest_checkpoint = None
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    # 搜尋 Checkpoint
    files = glob.glob(os.path.join(CHECKPOINT_PATH, "step_*.pt"))
    if files:
        try:
            files.sort(key=lambda f: int(re.search(r'step_(\d+)\.pt', f).group(1)), reverse=True)
            latest_checkpoint = files[0]
            step = int(re.search(r'step_(\d+)\.pt', latest_checkpoint).group(1))
            print(f"✅ 找到 Hybrid Checkpoint，將從 [Step {step}] 繼續...")
        except:
            step = 0

    # 初始化 Hybrid 模型 (Scalar Dim = 11)
    policy = HybridValueNet(scalar_dim=11).to(DEVICE)
    target = HybridValueNet(scalar_dim=11).to(DEVICE)
        
    if latest_checkpoint:
        try:
            policy.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE))
        except Exception as e:
            print(f"⚠️ 載入失敗: {e}。將重新開始。")
            step = 0
    else:
        print("ℹ️ 開始全新的 Hybrid 模型訓練。")
        
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimiz = optim.Adam(policy.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler('cuda')

    # Replay Buffer 格式: (cnn, scalar, reward, done, next_cnn, next_scalar)
    replay = deque(maxlen=BUFFER_SIZE)

    episode_scores = deque(maxlen=50)
    running_losses = deque(maxlen=1000)

    shot, piece, next_piece, current_episode_score = _env_reset()
    original_score = 0

    print(f"--- 開始訓練 (Hybrid Architecture) ---")

    while step < TRAIN_STEPS:
        moves = 0
        while moves < MAX_MOVES_PER_GAME and step < TRAIN_STEPS:
            legal_moves = _enumerate_legal_moves(shot, piece)
            if not legal_moves:
                break

            eps = epsilon_by_step(step)
            
            # Epsilon-Greedy
            if random.random() < eps:
                action = random.choice(legal_moves)
                next_shot, reward, done, new_next_piece = _simulate_step(shot, piece, action)
            else:
                # 預測最佳動作
                # 需要把所有可能動作的 (NextState, NextPiece) 丟進網路評估
                cnn_list, scalar_list, results = [], [], []
                
                # 下個方塊的形狀 (用於 One-Hot)
                nxt_shape = next_piece.shape

                for a in legal_moves:
                    sim_s, r, d, sim_next_p = _simulate_step(shot, piece, a)
                    # 這裡我們要評估的是 "sim_s" 這個狀態的好壞
                    # 而這個狀態面對的 "下一個方塊" 是 nxt_shape (也就是目前的 next_piece)
                    # 注意: 這裡有點繞。Q(S, A) = R + gamma * V(S')
                    # 我們的 ValueNet 評估的是 V(S')。
                    # 但這裡我們是在挑選動作 A，所以我們要比較的是 V(S_after_move)。
                    # S_after_move 包含了: 放置後的盤面 + 接下來要下的方塊(next_piece)。
                    
                    c_in, s_in = get_hybrid_input(sim_s.status, nxt_shape)
                    cnn_list.append(c_in)
                    scalar_list.append(s_in)
                    results.append((a, sim_s, r, d, sim_next_p))

                # Batch Predict
                with torch.no_grad(), autocast():
                    b_c = torch.tensor(np.array(cnn_list), dtype=torch.float32, device=DEVICE)
                    b_s = torch.tensor(np.array(scalar_list), dtype=torch.float32, device=DEVICE)
                    
                    # 預測價值
                    v_preds = policy(b_c, b_s).squeeze(-1) # (N,)
                    
                    # Q = R + gamma * V_next? 
                    # 不，我們是一個 Direct Value Agent。我們直接評估 Action 後狀態的價值。
                    # 所以 Score = Reward + GAMMA * V(State_After_Action)
                    # 但這裡的 Reward 是 transition reward。
                    
                    # 簡單化：直接選 V(S') 最高的操作，並加上當前的立即獎勵
                    # 因為 V(S') 代表未來的期望總分
                    rewards_t = torch.tensor([res[2] for res in results], device=DEVICE)
                    q_values = (rewards_t / REWARD_SCALE) + GAMMA * v_preds
                    
                    best_idx = torch.argmax(q_values).item()
                
                action, next_shot, reward, done, new_next_piece = results[best_idx]

            current_episode_score += reward
            original_score += (next_shot.score - shot.score)

            # 儲存到 Replay Buffer
            # S: (shot, piece) -> Action -> S': (next_shot, new_next_piece)
            # 我們存的是: Input(S), Reward, Done, Input(S')
            # Input(S) 其實是 (Shot狀態, NextPiece形狀)
            # Input(S') 其實是 (NextShot狀態, NewNextPiece形狀)
            
            curr_c, curr_s = get_hybrid_input(shot.status, piece.shape) # 這一步其實有點怪，通常 V(S) 不含 current piece。
            # 修正：我們的網路是評估「盤面 + 下個方塊 (Pending Piece)」。
            # 當前的決策點 S 是：盤面 shot + 手上的 piece + 預告的 next_piece。
            # 但 Value Network 通常評估的是 "穩態"。
            # 讓我們保持一致：
            # Store Transition:
            #   State: (Current Board, Next Piece Shape if we were to eval it... wait)
            #   比較標準的 DQN: S -> A -> R, S'
            #   S': (Next Board, New Next Piece Shape)
            #   S:  (Current Board, Current Piece Shape ?? No)
            
            # V4 邏輯修正：
            # 我們訓練 policy 去預測 Q-target。
            # State 是 (Board, Pending Piece)。
            # 動作執行後變成 (New Board, New Pending Piece)。
            
            s_c, s_s = get_hybrid_input(shot.status, piece.shape) # 這裡用 piece 當 pending
            next_c, next_s = get_hybrid_input(next_shot.status, new_next_piece.shape) # 這裡用 new_next 當 pending
            
            replay.append((s_c, s_s, reward, done, next_c, next_s))

            shot = next_shot
            piece = next_piece
            next_piece = new_next_piece
            moves += 1
            step += 1

            # 訓練步驟
            if len(replay) > BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)
                # unzip
                b_sc, b_ss, b_r, b_d, b_nc, b_ns = zip(*batch)
                
                t_sc = torch.tensor(np.array(b_sc), dtype=torch.float32, device=DEVICE)
                t_ss = torch.tensor(np.array(b_ss), dtype=torch.float32, device=DEVICE)
                t_r  = torch.tensor(b_r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                t_d  = torch.tensor(b_d, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                t_nc = torch.tensor(np.array(b_nc), dtype=torch.float32, device=DEVICE)
                t_ns = torch.tensor(np.array(b_ns), dtype=torch.float32, device=DEVICE)
                
                # Double DQN Logic / Target Net
                with torch.no_grad():
                    # V_next = Target(Next_State)
                    v_next = target(t_nc, t_ns)
                    y_target = (t_r / REWARD_SCALE) + GAMMA * v_next * (1 - t_d)
                
                with autocast():
                    v_pred = policy(t_sc, t_ss)
                    loss = loss_fn(v_pred, y_target)
                
                scaler.scale(loss).backward()
                scaler.step(optimiz)
                scaler.update()
                optimiz.zero_grad()
                
                running_losses.append(loss.item())

            if step % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())
            
            if step % LOG_INTERVAL == 0:
                avg_loss = np.mean(running_losses) if running_losses else 0
                avg_sc = np.mean(episode_scores) if episode_scores else 0
                print(f"[Step {step}] Loss={avg_loss:.5f} | AvgScore={avg_sc:.1f} | Eps={eps:.3f}")
                if len(episode_scores) > 0:
                    max_score = max(episode_scores)
                    print(f"          ↳ [Monitor] Recent max score: {max_score:.2f}")
                if step % CHECKPOINT_INTERVAL == 0:
                    path = os.path.join(CHECKPOINT_PATH, f"step_{step}.pt")
                    torch.save(policy.state_dict(), path)
                    print(f"Saved checkpoint: {path}")

            if done:
                break

        episode_scores.append(original_score)
        shot, piece, next_piece, current_episode_score = _env_reset()
        original_score = 0

    torch.save(policy.state_dict(), SAVE_PATH)
    print("訓練完成。")

if __name__ == '__main__':
    train()