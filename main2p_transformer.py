import pygame as pg
import pieces
import shots
import config
import Handler
import random
import copy
import numpy as np
import tetris_env
import torch
import torch.nn as nn
import math
import os

# 引入 Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

DEBUG = False
init_start = (5, 0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ppo_transformer_tetris_continued.zip"

# ------------------------------------------------------
# 1. Transformer 模型定義
# ------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

class TetrisTransformer(nn.Module):
    def __init__(self, board_dim: int = 200, n_pieces: int = 7, d_model: int = 128, nhead: int = 4, num_layers: int = 3, action_dim: int = 64):
        super().__init__()
        self.board_proj = nn.Linear(board_dim, d_model)
        self.piece_emb = nn.Embedding(n_pieces, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, board_flat: torch.Tensor, piece_id: torch.Tensor) -> torch.Tensor:
        board_token = self.board_proj(board_flat)
        piece_token = self.piece_emb(piece_id)
        tokens = torch.stack([piece_token, board_token], dim=0)
        tokens = self.pos_encoder(tokens)
        output = self.transformer(tokens)
        cls_token = output[0]
        return cls_token

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.transformer = TetrisTransformer(
            board_dim=200, n_pieces=7, d_model=128,
            nhead=4, num_layers=3, action_dim=64
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        piece_id = observations[:, 0].long()
        board_flat = observations[:, 1:]
        return self.transformer(board_flat, piece_id)

# ------------------------------------------------------
# 2. 載入與推論邏輯 (get_ppo_move)
# ------------------------------------------------------

def decode_action(action_id: int, max_rot: int = 4, min_x: int = -2, max_x: int = None):
    if max_x is None:
        max_x = config.columns + 3
    num_x = max_x - min_x + 1
    rot = action_id // num_x
    x_idx = action_id % num_x
    x = x_idx + min_x
    return x, rot

def load_ai_model():
    print(f"🔄 正在載入 PPO 模型: {MODEL_PATH} ...")
    if os.path.exists(MODEL_PATH):
        try:
            model = PPO.load(MODEL_PATH, device=DEVICE)
            print("✅ PPO 模型載入成功！")
            return model
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            return None
    else:
        print(f"⚠️ 找不到檔案 {MODEL_PATH}")
        return None

def get_ppo_move(model, shot, piece):
    """
    取得 AI 建議的動作 (target_x, target_rot)
    如果模型預測無效或無法移動，回傳 None
    """
    if model is None: return None

    # 1. 獲取乾淨盤面
    raw_board = np.array(shot.status, dtype=int)
    # 注意：這裡要確保跟訓練時的輸入格式一致。通常訓練時只看 0 和 1 (有無方塊)
    clean_board = (raw_board == 2).astype(np.float32).flatten()
    
    if len(clean_board) != 200: return None

    # 2. 準備 piece_id
    shape_list = list(config.shapes.keys())
    try: piece_id = shape_list.index(piece.shape)
    except: piece_id = 0
    
    obs = np.concatenate(([piece_id], clean_board))

    # 3. PPO 預測
    try:
        action, _ = model.predict(obs, deterministic=True)
        action_id = action.item() if isinstance(action, np.ndarray) else action
    except Exception as e:
        print(f"PPO Predict Error: {e}")
        return None

    # 4. 解碼與物理修正
    target_x, target_rot = decode_action(action_id)
    
    # 建立模擬環境來測試這個動作是否合法
    env = tetris_env.TetrisEnv()
    env.board = (raw_board == 2).astype(int)
    
    sim_piece = copy.deepcopy(piece)
    sim_piece.x = target_x
    sim_piece.rotation = target_rot
    sim_piece.y = 0 # 從頂部開始測試

    # 測試目標位置是否合法
    if not env._is_valid_position(env.board, sim_piece):
        # 嘗試簡單的 Wall Kick (左右微調)
        for offset in [0, -1, 1, -2, 2]:
            sim_piece.x = target_x + offset
            if env._is_valid_position(env.board, sim_piece):
                target_x += offset
                break
        else:
            # 所有嘗試都失敗，回傳 None
            return None
            
    return target_x, target_rot

# ------------------------------------------------------
# 3. 輔助函式
# ------------------------------------------------------

def getRandomPiece():
    shape = random.choice(list(config.shapes.keys()))
    piece = pieces.Piece(*init_start, shape)
    return piece

def draw_grid(surface, offset_x):
    grid_surface = pg.Surface((config.columns * config.grid, config.rows * config.grid), pg.SRCALPHA)
    color = (150, 150, 150, 60)
    for y in range(config.rows):
        pg.draw.line(grid_surface, color, (0, y * config.grid), (config.columns * config.grid, y * config.grid))
    for x in range(config.columns):
        pg.draw.line(grid_surface, color, (x * config.grid, 0), (x * config.grid, config.rows * config.grid))
    surface.blit(grid_surface, (offset_x, 0))

def draw_player_ui(screen, shot, piece, next_piece, font, offset_x, score_pos, line_pos, next_piece_pos, garbage_bar_pos):
    # 繪製背景與固定方塊
    for y in range(config.rows):
        for x in range(config.columns):
            # 繪製背景格
            pg.draw.rect(screen, shot.color[y][x], (offset_x + x * config.grid, y * config.grid, config.grid, config.grid))
            # 繪製邊框
            if shot.status[y][x] == 0:
                pg.draw.rect(screen, (30, 30, 30), (offset_x + x * config.grid, y * config.grid, config.grid, config.grid), 1)

    # 繪製當前移動中的方塊
    if not piece.is_fixed:
        for y, x in Handler.getCellsAbsolutePosition(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                pg.draw.rect(screen, piece.color, (offset_x + x * config.grid, y * config.grid, config.grid, config.grid))

    # 文字資訊
    textsurface = font.render(f'Score: {shot.score}', False, (255, 255, 255))
    screen.blit(textsurface, score_pos)
    textsurface = font.render(f'Line: {shot.line_count}', False, (255, 255, 255))
    screen.blit(textsurface, line_pos)

    # Next Piece
    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (next_piece_pos[0] + x * config.grid, next_piece_pos[1] + y * config.grid, config.grid, config.grid))
    for y, x in next_piece.getCells():
        pg.draw.rect(screen, next_piece.color, (next_piece_pos[0] + x * config.grid, next_piece_pos[1] + y * config.grid, config.grid, config.grid))

    # Garbage Bar
    if shot.pending_garbage > 0:
        bar_h = config.height * 0.9
        fill_ratio = min(shot.pending_garbage, 12) / 12.0
        fill_h = bar_h * fill_ratio
        bx, by = garbage_bar_pos[0], config.height * 0.05
        pg.draw.rect(screen, (80, 80, 80), (bx, by, config.GARBAGE_BAR_WIDTH, bar_h))
        pg.draw.rect(screen, (255, 50, 50), (bx, (by + bar_h) - fill_h, config.GARBAGE_BAR_WIDTH, fill_h))

    draw_grid(screen, offset_x)

# ------------------------------------------------------
# 4. 主程式
# ------------------------------------------------------

def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris 1v1: Human vs PPO Transformer")

    ai_model = load_ai_model()

    shot1, piece1, next_piece1 = shots.Shot(), getRandomPiece(), getRandomPiece()
    shot2, piece2, next_piece2 = shots.Shot(), getRandomPiece(), getRandomPiece()

    # 兩邊的自動下落計時器
    counter1 = 0
    counter2 = 0
    
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    
    game_over1 = False
    game_over2 = False

    # --- AI First Move Logic ---
    # 遊戲一開始，先讓 AI 做一次決策，避免它發呆
    if not game_over2 and ai_model is not None:
        move = get_ppo_move(ai_model, shot2, piece2)
        if move is not None:
            # 如果有決策，就瞬間移動到位
            piece2.x, piece2.rotation = move
            Handler.instantDrop(shot2, piece2)
        else:
            # 如果沒有決策，就讓它原地落下
            print("AI Start Fallback: No move found, dropping.")
            Handler.instantDrop(shot2, piece2)

    run = True
    while run:
        # ---------------------------------------
        # 1. Human Player (P1) Logic
        # ---------------------------------------
        if not DEBUG and not game_over1:
            if counter1 == config.difficulty:
                Handler.drop(shot1, piece1)
                counter1 = 0
            else:
                counter1 += 1
        
        # P1 Input Handling
        for event in pg.event.get():
            if event.type == pg.QUIT: run = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE: run = False
                
                if not game_over1:
                    if event.key == pg.K_w: Handler.rotate(shot1, piece1)
                    if event.key == pg.K_s: key_ticker1[pg.K_s] = 13; Handler.drop(shot1, piece1)
                    if event.key == pg.K_a: key_ticker1[pg.K_a] = 13; Handler.moveLeft(shot1, piece1)
                    if event.key == pg.K_d: key_ticker1[pg.K_d] = 13; Handler.moveRight(shot1, piece1)
                    if event.key == pg.K_LSHIFT: Handler.instantDrop(shot1, piece1)

        keys = pg.key.get_pressed()
        if not game_over1:
            if keys[pg.K_a] and key_ticker1[pg.K_a] == 0: key_ticker1[pg.K_a] = 6; Handler.moveLeft(shot1, piece1)
            if keys[pg.K_d] and key_ticker1[pg.K_d] == 0: key_ticker1[pg.K_d] = 6; Handler.moveRight(shot1, piece1)
            if keys[pg.K_s] and key_ticker1[pg.K_s] == 0: key_ticker1[pg.K_s] = 6; Handler.drop(shot1, piece1)
        
        for k in key_ticker1:
            if key_ticker1[k] > 0: key_ticker1[k] -= 1

        # ---------------------------------------
        # 2. AI Player (P2) Logic
        # ---------------------------------------
        
        # [修正] AI 的自動下落 (Fallback)
        # 如果 AI 卡住了，或者沒有執行 instantDrop，這裡確保方塊會因為重力掉下來
        if not game_over2 and not piece2.is_fixed:
            if counter2 == config.difficulty: # 這裡可以用更快的速度，例如 config.difficulty // 2
                Handler.drop(shot2, piece2)
                counter2 = 0
            else:
                counter2 += 1

        # ---------------------------------------
        # 3. Game State Updates
        # ---------------------------------------

        # Garbage Timer
        for shot, p, go in [(shot1, piece1, game_over1), (shot2, piece2, game_over2)]:
            if not go and shot.pending_garbage > 0:
                shot.garbage_insert_timer += 1
                if shot.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                    lines = min(shot.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                    Handler.insertGarbage(shot, lines)
                    shot.pending_garbage -= lines
                    shot.garbage_insert_timer = 0

        # P1 Line Clear & New Piece
        if not game_over1 and piece1.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot1, piece1)
            atk1 = Handler.calculateAttack(clears, shot1.combo_count, shot1.is_b2b, all_clear)
            if clears > 0: shot1.combo_count += 1; shot1.is_b2b = (clears == 4)
            else: shot1.combo_count = 0
            
            if atk1 > 0:
                cancel = min(atk1, shot1.pending_garbage)
                shot1.pending_garbage -= cancel
                atk1 -= cancel
                shot2.pending_garbage += atk1
            
            piece1, next_piece1 = next_piece1, getRandomPiece()
            if Handler.isDefeat(shot1, piece1):
                game_over1 = True
                print("P1 Game Over")

        # P2 Line Clear & New Piece
        if not game_over2 and piece2.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot2, piece2)
            atk2 = Handler.calculateAttack(clears, shot2.combo_count, shot2.is_b2b, all_clear)
            if clears > 0: shot2.combo_count += 1; shot2.is_b2b = (clears == 4)
            else: shot2.combo_count = 0
            
            if atk2 > 0:
                cancel = min(atk2, shot2.pending_garbage)
                shot2.pending_garbage -= cancel
                atk2 -= cancel
                shot1.pending_garbage += atk2

            piece2, next_piece2 = next_piece2, getRandomPiece()
            
            # 檢查生成後是否立刻死亡
            if Handler.isDefeat(shot2, piece2):
                game_over2 = True
                print("P2 Game Over")
            else:
                # 只要沒死，AI 就要做決策
                if ai_model is not None:
                    move = get_ppo_move(ai_model, shot2, piece2)
                    if move is not None:
                        # 決策成功：執行動作並落下
                        piece2.x, piece2.rotation = move
                        Handler.instantDrop(shot2, piece2)
                    else:
                        # 決策失敗 (None)：讓它原地落下
                        # 這樣可以避免卡在空中，至少遊戲會繼續進行
                        print("AI Move Failed, falling naturally.")
                        Handler.instantDrop(shot2, piece2) 

        if game_over1 and game_over2:
            run = False

        screen.fill(config.background_color)
        if not game_over1: draw_player_ui(screen, shot1, piece1, next_piece1, myfont, config.P1_OFFSET_X, config.P1_SCORE_POS, config.P1_LINE_POS, config.P1_NEXT_PIECE_POS, config.P1_GARBAGE_BAR_POS)
        if not game_over2: draw_player_ui(screen, shot2, piece2, next_piece2, myfont, config.P2_OFFSET_X, config.P2_SCORE_POS, config.P2_LINE_POS, config.P2_NEXT_PIECE_POS, config.P2_GARBAGE_BAR_POS)
        
        pg.display.update()
        fpsClock.tick(config.fps)

    print("----- Final Result -----")
    print(f"P1 Score: {shot1.score} | Lines: {shot1.line_count}")
    print(f"P2 Score: {shot2.score} | Lines: {shot2.line_count}")
    pg.quit()

if __name__ == "__main__":
    main()
