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
import numpy as np
import math
import os
from dataset import decode_action  # 用來把 action_id 還原成 (x, rot)
DEBUG = False
init_start = (5, 0) 

# --- 核心設定：8-Feature Tetris AI 權重 ---
# 特徵順序: [Landing, RowTrans, ColTrans, Holes, WellSums, DeepWells, CumWells, MaxHeight]
# 這裡填入你 CMA-ES 訓練出來的 Top Weights
# 如果還沒跑完，這是一組強力的手動調整版 (鼓勵 Tetris):
# DeepWells 是正的 (+0.5) 代表鼓勵留深坑

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
    def __init__(
        self,
        board_dim: int = 200,
        n_pieces: int = 7,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        action_dim: int = 64
    ):
        super().__init__()
        self.board_proj = nn.Linear(board_dim, d_model)
        self.piece_emb = nn.Embedding(n_pieces, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, board_flat: torch.Tensor, piece_id: torch.Tensor) -> torch.Tensor:
        # board_flat: (batch, 200)
        # piece_id:   (batch,)
        board_token = self.board_proj(board_flat)     # (batch, d_model)
        piece_token = self.piece_emb(piece_id)        # (batch, d_model)

        tokens = torch.stack([piece_token, board_token], dim=0)  # (seq=2, batch, d_model)
        tokens = self.pos_encoder(tokens)                         # 加位置編碼

        output = self.transformer(tokens)        # (seq=2, batch, d_model)
        cls_token = output[0]                    # (batch, d_model)
        logits = self.action_head(cls_token)     # (batch, action_dim)
        return logits

MODEL_PATH = "transformer_tetris.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ai_model():
    model = TetrisTransformer().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        print("✅ Transformer 模型載入成功")
        return model
    else:
        print("⚠️ 找不到模型檔案，AI 會停用")
        return None

def get_transformer_move(model, shot, piece):
    if model is None:
        return None

    # 盤面轉成 0/1 mask，跟訓練時一致
    board_np = (np.array(shot.status, dtype=np.int32) == 2).astype(np.float32)
    board_flat = board_np.reshape(1, -1)                      # (1, 200)

    # 方塊 ID：對應 config.shapes 的索引
    shape_list = list(config.shapes.keys())
    piece_id = shape_list.index(piece.shape)

    board_t = torch.tensor(board_flat, dtype=torch.float32).to(DEVICE)
    piece_t = torch.tensor([piece_id], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(board_t, piece_t)                      # (1, 64)
        action_id = logits.argmax(dim=1).item()

    x, rot = decode_action(action_id)

    # 簡單保護一下，避免非法 x 直接炸遊戲
    if x < -2 or x > config.columns + 3:
        return None

    return x, rot


def get_ai_move_heuristic(shot, piece):
    """
    使用 8-Feature 演算法決定最佳移動
    """
    env = tetris_env.TetrisEnv()
    env.board = np.array(shot.status, dtype=int)
    env.current_piece = copy.deepcopy(piece)
    
    possible_moves = {}
    piece = env.current_piece
    num_rotations = len(config.shapes[piece.shape])
    
    for rot in range(num_rotations):
        for x in range(-2, config.columns + 1):
            sim_piece = copy.deepcopy(piece)
            sim_piece.rotation = rot
            sim_piece.x = x
            sim_piece.y = 0 
            
            if not env._is_valid_position(env.board, sim_piece):
                continue
            
            while env._is_valid_position(env.board, sim_piece, adj_x=0, adj_y=1):
                sim_piece.y += 1
            
            temp_board = env.board.copy()
            env._lock_piece(temp_board, sim_piece)
            possible_moves[(x, rot)] = temp_board

    if not possible_moves:
        return None 
        
    best_score = -float('inf')
    best_move = None
    
    for move, board_state in possible_moves.items():
        # 使用新的 8 參數特徵計算
        features = get_tetris_features_v8(board_state)
        score = np.dot(BEST_WEIGHTS, features)
        
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move 

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
    for y in range(config.rows):
        for x in range(config.columns):
            if shot.status[y][x] != 2:
                shot.status[y][x] = 0

    if not piece.is_fixed:
        for y, x in Handler.getCellsAbsolutePosition(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                shot.color[y][x] = piece.color
                shot.status[y][x] = 1

    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            if shot.status[y][x] == 0:
                draw_color = (0, 0, 0)
            else:
                draw_color = color
            pg.draw.rect(screen, draw_color, (offset_x + x * config.grid, y * config.grid, config.grid, config.grid))

    textsurface = font.render(f'Score: {shot.score}', False, (255, 255, 255))
    screen.blit(textsurface, score_pos)
    textsurface = font.render(f'Line: {shot.line_count}', False, (255, 255, 255))
    screen.blit(textsurface, line_pos)

    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (next_piece_pos[0] + x * config.grid, next_piece_pos[1] + y * config.grid, config.grid, config.grid))

    for y, x in next_piece.getCells():
        color = next_piece.color
        pg.draw.rect(screen, color, (next_piece_pos[0] + x * config.grid, next_piece_pos[1] + y * config.grid, config.grid, config.grid))
    
    if shot.pending_garbage > 0:
        bar_max_height = config.height * 0.9 
        bar_y_start = config.height * 0.05
        pending_visual = min(shot.pending_garbage, 12) 
        bar_fill_ratio = pending_visual / 12.0
        bar_height = bar_max_height * bar_fill_ratio
        bar_x = garbage_bar_pos[0]
        bar_y_fill = (bar_y_start + bar_max_height) - bar_height
        
        pg.draw.rect(screen, (80, 80, 80), (bar_x, bar_y_start, config.GARBAGE_BAR_WIDTH, bar_max_height))
        pg.draw.rect(screen, (255, 50, 50), (bar_x, bar_y_fill, config.GARBAGE_BAR_WIDTH, bar_height))

    draw_grid(screen, offset_x)

def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris 1v1: Human vs 8-Feature AI")

    # --- P1 (Human) ---
    shot1 = shots.Shot()
    piece1 = getRandomPiece()
    next_piece1 = getRandomPiece()
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    game_over1 = False

    # --- P2 (AI) ---
    ai_model = load_ai_model()

    # --- P2 遊戲狀態 (AI) ---
    shot2 = shots.Shot()
    piece2 = getRandomPiece()
    next_piece2 = getRandomPiece()
    game_over2 = False
        
    # AI First Move
    if ai_model is not None and not game_over2:
        move = get_transformer_move(ai_model, shot2, piece2)
        if move is not None:
            piece2.x, piece2.rotation = move
        Handler.instantDrop(shot2, piece2)
    
    run = True
    while run:
        # Auto Drop (Human only)
        if not DEBUG and not game_over1:
            if counter1 == config.difficulty:
                Handler.drop(shot1, piece1)
                counter1 = 0
            else:
                counter1 += 1

        # Garbage Handling
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

        # Events
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

        # P1 Update
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
            if Handler.isDefeat(shot1, piece1): game_over1 = True; print("P1 Game Over")

        # P2 (AI) Update
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
            
            # AI Think
            if not game_over2 and ai_model is not None:
                move = get_transformer_move(ai_model, shot2, piece2)
                if move is not None:
                    piece2.x, piece2.rotation = move
                Handler.instantDrop(shot2, piece2)

        if game_over1 and game_over2: run = False
        
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
