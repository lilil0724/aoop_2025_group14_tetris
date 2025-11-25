import pygame as pg
import torch
import torch.nn as nn
import numpy as np
import math
import os
import copy
import random

# 引入 SB3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

# 你的遊戲邏輯檔案
import pieces
import shots
import config
import Handler
import tetris_env # 用來取得 observation

# 如果你的 dataset.py 有這一行，可以直接 import；沒有的話就用下面的函式
from dataset import decode_action 

DEBUG = False
init_start = (5, 0)

# ------------------------------------------------------
# 1. 必須重現訓練時的模型結構 (Transformer + Extractor)
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
        return cls_token # 注意：PPO 不用這裡的 logits，只拿特徵

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
# 2. 載入與推論
# ------------------------------------------------------

MODEL_PATH = "ppo_transformer_tetris_continued.zip" # 你的模型檔名
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ppo_model():
    print(f"🔄 正在載入 PPO 模型: {MODEL_PATH} ...")
    if os.path.exists(MODEL_PATH):
        # 載入 PPO 模型
        # custom_objects 告訴 PPO 我們的 Extractor 類別在哪裡
        try:
            model = PPO.load(MODEL_PATH, device=DEVICE)
            print("✅ PPO 模型載入成功！準備戰鬥！")
            return model
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            print("可能原因：類別定義不一致，或缺少 stable-baselines3")
            return None
    else:
        print(f"⚠️ 找不到檔案 {MODEL_PATH}")
        return None

def get_ppo_move(model, shot, piece):
    if model is None:
        return None

    # 1. 準備 observation (跟訓練時一樣：[piece_id, ...board...])
    # 盤面轉成 0/1
    board_np = (np.array(shot.status) == 2).astype(np.float32).flatten()
    
    shape_list = list(config.shapes.keys())
    piece_id = shape_list.index(piece.shape)
    
    obs = np.concatenate(([piece_id], board_np))
    
    # 2. 預測動作
    # predict 回傳 (action, state)，我們只要 action
    # deterministic=True 代表不使用隨機探索，直接選機率最高的
    action_id, _ = model.predict(obs, deterministic=True)
    
    # action_id 是一個 numpy array 或 int
    if isinstance(action_id, np.ndarray):
        action_id = action_id.item()
        
    # 3. 解碼
    x, rot = decode_action(action_id)
    
    # 保護
    if x < -2 or x > config.columns + 3:
        return None
        
    return x, rot

# ------------------------------------------------------
# 3. 遊戲主程式 (只修改 AI 部分)
# ------------------------------------------------------

def getRandomPiece():
    shape = random.choice(list(config.shapes.keys()))
    return pieces.Piece(*init_start, shape)

# (省略 draw_grid 和 draw_player_ui，這兩者與你原本的完全一樣)
def draw_grid(surface, offset_x):
    grid_surface = pg.Surface((config.columns * config.grid, config.rows * config.grid), pg.SRCALPHA)
    color = (150, 150, 150, 60)
    for y in range(config.rows):
        pg.draw.line(grid_surface, color, (0, y * config.grid), (config.columns * config.grid, y * config.grid))
    for x in range(config.columns):
        pg.draw.line(grid_surface, color, (x * config.grid, 0), (x * config.grid, config.rows * config.grid))
    surface.blit(grid_surface, (offset_x, 0))

def draw_player_ui(screen, shot, piece, next_piece, font, offset_x, score_pos, line_pos, next_piece_pos, garbage_bar_pos): 
    for y in range(config.rows):
        for x in range(config.columns):
            if shot.status[y][x] != 2: shot.status[y][x] = 0
    if not piece.is_fixed:
        for y, x in Handler.getCellsAbsolutePosition(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                shot.color[y][x] = piece.color
                shot.status[y][x] = 1
    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            draw_color = color if shot.status[y][x] != 0 else (0, 0, 0)
            pg.draw.rect(screen, draw_color, (offset_x + x * config.grid, y * config.grid, config.grid, config.grid))
    textsurface = font.render(f'Score: {shot.score}', False, (255, 255, 255))
    screen.blit(textsurface, score_pos)
    textsurface = font.render(f'Line: {shot.line_count}', False, (255, 255, 255))
    screen.blit(textsurface, line_pos)
    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (next_piece_pos[0] + x * config.grid, next_piece_pos[1] + y * config.grid, config.grid, config.grid))
    for y, x in next_piece.getCells():
        pg.draw.rect(screen, next_piece.color, (next_piece_pos[0] + x * config.grid, next_piece_pos[1] + y * config.grid, config.grid, config.grid))
    if shot.pending_garbage > 0:
        bar_h = config.height * 0.9 
        fill_ratio = min(shot.pending_garbage, 12) / 12.0
        fill_h = bar_h * fill_ratio
        bx, by = garbage_bar_pos[0], config.height * 0.05
        pg.draw.rect(screen, (80, 80, 80), (bx, by, config.GARBAGE_BAR_WIDTH, bar_h))
        pg.draw.rect(screen, (255, 50, 50), (bx, (by + bar_h) - fill_h, config.GARBAGE_BAR_WIDTH, fill_h))
    draw_grid(screen, offset_x)

def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris 1v1: Human vs PPO Transformer")

    # 載入 PPO 模型
    ai_model = load_ppo_model()

    # 遊戲初始化
    shot1, piece1, next_piece1 = shots.Shot(), getRandomPiece(), getRandomPiece()
    shot2, piece2, next_piece2 = shots.Shot(), getRandomPiece(), getRandomPiece()
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    game_over1 = False
    game_over2 = False
    
    # AI First Move
    if ai_model is not None and not game_over2:
        move = get_ppo_move(ai_model, shot2, piece2)
        if move is not None:
            piece2.x, piece2.rotation = move
        Handler.instantDrop(shot2, piece2)
    
    run = True
    while run:
        # P1 Logic
        if not DEBUG and not game_over1:
            if counter1 == config.difficulty:
                Handler.drop(shot1, piece1)
                counter1 = 0
            else:
                counter1 += 1

        if not game_over1 and shot1.pending_garbage > 0:
            shot1.garbage_insert_timer += 1
            if shot1.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                lines = min(shot1.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                Handler.insertGarbage(shot1, lines)
                shot1.pending_garbage -= lines
                shot1.garbage_insert_timer = 0
                if Handler.isDefeat(shot1, piece1): game_over1 = True

        if not game_over2 and shot2.pending_garbage > 0:
            shot2.garbage_insert_timer += 1
            if shot2.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                lines = min(shot2.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                Handler.insertGarbage(shot2, lines)
                shot2.pending_garbage -= lines
                shot2.garbage_insert_timer = 0
                if Handler.isDefeat(shot2, piece2): game_over2 = True

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
            if Handler.isDefeat(shot1, piece1): game_over1 = True

        # P2 Update (AI)
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
            if Handler.isDefeat(shot2, piece2): game_over2 = True; print("P2 Game Over")
            
            # AI 思考
            if not game_over2 and ai_model is not None:
                move = get_ppo_move(ai_model, shot2, piece2)
                if move is not None:
                    piece2.x, piece2.rotation = move
                Handler.instantDrop(shot2, piece2)

        if game_over1 and game_over2: run = False
        
        screen.fill(config.background_color)
        if not game_over1: draw_player_ui(screen, shot1, piece1, next_piece1, myfont, config.P1_OFFSET_X, config.P1_SCORE_POS, config.P1_LINE_POS, config.P1_NEXT_PIECE_POS, config.P1_GARBAGE_BAR_POS)
        if not game_over2: draw_player_ui(screen, shot2, piece2, next_piece2, myfont, config.P2_OFFSET_X, config.P2_SCORE_POS, config.P2_LINE_POS, config.P2_NEXT_PIECE_POS, config.P2_GARBAGE_BAR_POS)

        pg.display.update()
        fpsClock.tick(config.fps)
    
    pg.quit()

if __name__ == "__main__":
    main()
