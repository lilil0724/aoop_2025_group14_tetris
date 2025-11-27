import pygame as pg
import sys
import copy
import random
import threading
import socket

# 引用模組
import config
import pieces
import shots
import Handler
import numpy as np
import network_utils # [NEW] Network Module
try:
    import tetris_env
except ImportError:
    print("Warning: tetris_env.py not found. Heuristic AI might fail.")
# 嘗試匯入 AI
try:
    from ai_player_nn import AIPlayerNN
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: ai_player_nn.py not found. 1vAI mode will be disabled or random.")

# --- 全域設定變數 ---
AI_MOVE_DELAY = 5 
AI_THINKING_DELAY = 10 
SHOW_GHOST = True  # 預設開啟投影

# --- UI 元件 ---
class Button:
    def __init__(self, x, y, w, h, text, action_code, color=(50, 200, 50), hover_color=(100, 255, 100)):
        self.rect = pg.Rect(x, y, w, h)
        self.text = text
        self.action_code = action_code
        self.color = color
        self.hover_color = hover_color
        self.font = pg.font.SysFont('Arial', 30, bold=True)

    def draw(self, screen):
        mouse_pos = pg.mouse.get_pos()
        current_color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        
        x, y, w, h = self.rect.x, self.rect.y, self.rect.w, self.rect.h
        r, g, b = current_color
        
        # 亮部與暗部計算
        light = (min(255, int(r * 1.4)), min(255, int(g * 1.4)), min(255, int(b * 1.4)))
        dark = (int(r * 0.6), int(g * 0.6), int(b * 0.6))
        
        bevel = 6 # 邊框厚度
        
        # 1. 填滿中心
        pg.draw.rect(screen, current_color, (x + bevel, y + bevel, w - 2*bevel, h - 2*bevel))
        
        # 2. 繪製立體邊框 (梯形)
        # 上 (亮)
        pg.draw.polygon(screen, light, [(x, y), (x + w, y), (x + w - bevel, y + bevel), (x + bevel, y + bevel)])
        # 左 (亮)
        pg.draw.polygon(screen, light, [(x, y), (x + bevel, y + bevel), (x + bevel, y + h - bevel), (x, y + h)])
        # 下 (暗)
        pg.draw.polygon(screen, dark, [(x, y + h), (x + w, y + h), (x + w - bevel, y + h - bevel), (x + bevel, y + h - bevel)])
        # 右 (暗)
        pg.draw.polygon(screen, dark, [(x + w, y), (x + w, y + h), (x + w - bevel, y + h - bevel), (x + w - bevel, y + bevel)])
        
        # 3. 文字
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        
        # 按下時的位移效果 (視覺回饋)
        if self.rect.collidepoint(mouse_pos) and pg.mouse.get_pressed()[0]:
             text_rect.move_ip(2, 2)
             
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# --- 輔助功能: 投影位置計算 ---
def get_ghost_piece(shot, piece):
    """
    計算並回傳一個「投影方塊」，該方塊位於目前方塊正下方的著陸點。
    """
    ghost = copy.deepcopy(piece)
    while True:
        can_move_down = True
        for y, x in Handler.getCellsAbsolutePosition(ghost):
            ny = y + 1
            if ny >= config.rows:
                can_move_down = False
                break
            if ny >= 0 and shot.status[ny][x] == 2:
                can_move_down = False
                break
        
        if can_move_down:
            ghost.y += 1
        else:
            break
    return ghost

# --- 繪圖輔助 ---

# --- [新增] 移植自 main_2p.py 的啟發式 AI 核心 ---

# 特徵權重 (這就是 AI 的性格參數)
BEST_WEIGHTS = np.array([-1.41130507, -2.23926392, -0.78272467, -4.00369693, -0.67902086, -0.449347,
                         -0.1623215, -0.91940282])

def get_tetris_features_v8(board):
    """ 計算盤面特徵 (AI 的眼睛) """
    grid = (np.array(board) == 2).astype(int)
    rows, cols = grid.shape
    
    # 1. Landing Height
    row_indices = np.arange(rows, 0, -1).reshape(-1, 1)
    height_grid = grid * row_indices
    col_heights = np.max(height_grid, axis=0)
    landing_height = np.mean(col_heights)
    
    # 2. Row Transitions
    row_trans = 0
    for r in range(rows):
        line = np.insert(grid[r], [0, cols], 1)
        row_trans += np.sum(np.abs(np.diff(line)))
        
    # 3. Column Transitions
    col_trans = 0
    for c in range(cols):
        col = np.insert(grid[:, c], [0, rows], [0, 1])
        col_trans += np.sum(np.abs(np.diff(col)))
        
    # 4. Holes
    cumsum = np.cumsum(grid, axis=0)
    holes = np.sum((cumsum > 0) & (grid == 0))
    
    # 5. Well Analysis
    well_depths = []
    for c in range(cols):
        if c == 0: left_wall = np.ones(rows)
        else: left_wall = grid[:, c-1]
        if c == cols-1: right_wall = np.ones(rows)
        else: right_wall = grid[:, c+1]
        mid = grid[:, c]
        is_well = (left_wall == 1) & (right_wall == 1) & (mid == 0)
        depth = 0
        for r in range(rows):
            if is_well[r]: depth += 1
            else:
                if depth > 0: well_depths.append(depth)
                depth = 0
        if depth > 0: well_depths.append(depth)
        
    well_sums = sum(well_depths)
    deep_wells = sum([d for d in well_depths if d >= 3])
    cum_wells = sum([d*(d+1)/2 for d in well_depths])
    max_height = np.max(col_heights) if len(col_heights) > 0 else 0
    
    features = np.array([landing_height, row_trans, col_trans, holes, well_sums, deep_wells, cum_wells, max_height], dtype=np.float32)
    
    # 正規化
    features[0] /= 10.0
    features[1] /= 100.0
    features[2] /= 100.0
    features[3] /= 40.0
    features[4] /= 40.0
    features[5] /= 40.0
    features[6] /= 100.0
    features[7] /= 20.0
    return features

def get_ai_move_heuristic(shot, piece):
    """ 思考最佳移動路徑 (AI 的大腦) """
    # 建立模擬環境
    env = tetris_env.TetrisEnv()
    env.board = np.array(shot.status, dtype=int)
    env.current_piece = copy.deepcopy(piece)
    
    possible_moves = {}
    piece_ref = env.current_piece
    num_rotations = len(config.shapes[piece_ref.shape])
    
    # 窮舉所有可能的落點
    for rot in range(num_rotations):
        for x in range(-2, config.columns + 1):
            sim_piece = copy.deepcopy(piece_ref)
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
    
    # 評估每個落點的分數
    for move, board_state in possible_moves.items():
        features = get_tetris_features_v8(board_state)
        score = np.dot(BEST_WEIGHTS, features)
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move
# -------------------------------------------

def draw_3d_block(surface, color, x, y, size):
    """ 繪製立體方塊 (Bevel Effect) """
    r, g, b = color
    # 亮部 (Top/Left) - 提亮
    light = (min(255, int(r * 1.4)), min(255, int(g * 1.4)), min(255, int(b * 1.4)))
    # 暗部 (Bottom/Right) - 壓暗
    dark = (int(r * 0.6), int(g * 0.6), int(b * 0.6))
    
    bevel = size // 6  # 邊框厚度
    
    # 1. 填滿中心 (原色)
    pg.draw.rect(surface, color, (x + bevel, y + bevel, size - 2*bevel, size - 2*bevel))
    
    # 2. 四個梯形邊框
    # 上 (亮)
    pg.draw.polygon(surface, light, [(x, y), (x + size, y), (x + size - bevel, y + bevel), (x + bevel, y + bevel)])
    # 左 (亮)
    pg.draw.polygon(surface, light, [(x, y), (x + bevel, y + bevel), (x + bevel, y + size - bevel), (x, y + size)])
    # 下 (暗)
    pg.draw.polygon(surface, dark, [(x, y + size), (x + size, y + size), (x + size - bevel, y + size - bevel), (x + bevel, y + size - bevel)])
    # 右 (暗)
    pg.draw.polygon(surface, dark, [(x + size, y), (x + size, y + size), (x + size - bevel, y + size - bevel), (x + size - bevel, y + bevel)])


def draw_grid(surface, offset_x):
    grid_surface = pg.Surface(
        (config.columns * config.grid, config.rows * config.grid), pg.SRCALPHA
    )
    # 內部的網格線維持淡灰色
    color = (150, 150, 150, 60)
    for y in range(config.rows):
        pg.draw.line(grid_surface, color, (0, y * config.grid), (config.columns * config.grid, y * config.grid))
    for x in range(config.columns):
        pg.draw.line(grid_surface, color, (x * config.grid, 0), (x * config.grid, config.rows * config.grid))
    surface.blit(grid_surface, (offset_x, 0))

def draw_player_ui(screen, shot, piece, next_piece, font, 
                   base_offset_x, score_pos_offset, line_pos_offset, next_piece_pos_offset, 
                   garbage_bar_pos_offset, player_name="Player"): 
    
    # --- 特效處理: 畫面震動 ---
    offset_x = base_offset_x
    offset_y = 0
    
    if getattr(shot, 'shake_timer', 0) > 0:
        offset_x += random.randint(-4, 4)
        offset_y += random.randint(-4, 4)
        shot.shake_timer -= 1
        # 震動時紅框警示
        border_rect = pg.Rect(offset_x - 2, offset_y - 2, config.columns * config.grid + 4, config.rows * config.grid + 4)
        pg.draw.rect(screen, (255, 50, 50), border_rect, 4)

    # 1. 繪製盤面 (已固定方塊)
    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            if shot.status[y][x] != 0:
                draw_3d_block(screen, color, 
                    offset_x + x * config.grid,
                    offset_y + y * config.grid,
                    config.grid
                )

    # 2. 繪製 [投影 Ghost Piece]
    global SHOW_GHOST
    if SHOW_GHOST and not piece.is_fixed:
        ghost = get_ghost_piece(shot, piece)
        for y, x in Handler.getCellsAbsolutePosition(ghost):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                # 深灰色細框
                pg.draw.rect(screen, (80, 80, 80), (
                    offset_x + x * config.grid,
                    offset_y + y * config.grid,
                    config.grid,
                    config.grid
                ), 1) 

    # 3. 繪製 [移動中的實體方塊]
    if not piece.is_fixed:
        for y, x in Handler.getCellsAbsolutePosition(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                draw_3d_block(screen, piece.color,
                    offset_x + x * config.grid,
                    offset_y + y * config.grid,
                    config.grid
                )

    # 4. Grid (內部網格)
    draw_grid(screen, offset_x)

    # 5. [NEW] 繪製明顯的外框邊界
    # 使用白色 (255, 255, 255)，線寬 3
    pg.draw.rect(screen, (255, 255, 255), (
        offset_x, 
        offset_y, 
        config.columns * config.grid, 
        config.rows * config.grid
    ), 3)
    
    # --- UI 資訊區 ---
    info_start_x = offset_x + (config.columns * config.grid) + 20 
    
    label_font = pg.font.SysFont('Arial', 20)
    name_surf = label_font.render(player_name, True, (200, 200, 200))
    screen.blit(name_surf, (offset_x, -30))

    textsurface = font.render(f'Score: {shot.score}', False, (255, 255, 255))
    screen.blit(textsurface, (info_start_x, config.rows * config.grid // 2))

    textsurface = font.render(f'Line: {shot.line_count}', False, (255, 255, 255))
    screen.blit(textsurface, (info_start_x, config.rows * config.grid // 2 + 50))

    # Next Piece
    next_center_x = info_start_x + 60
    next_center_y = config.rows * config.grid // 2 - 150
    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (next_center_x + x * config.grid, next_center_y + y * config.grid, config.grid, config.grid))
    for y, x in next_piece.getCells():
        draw_3d_block(screen, next_piece.color, next_center_x + x * config.grid, next_center_y + y * config.grid, config.grid)

    # Garbage Bar
    if shot.pending_garbage > 0:
        bar_max_height = config.height * 0.9
        bar_y_start = config.height * 0.05
        pending_visual = min(shot.pending_garbage, 20) 
        bar_fill_ratio = pending_visual / 20.0
        bar_height = bar_max_height * bar_fill_ratio
        bar_x = offset_x - config.GARBAGE_BAR_WIDTH - 5
        bar_y_fill = (bar_y_start + bar_max_height) - bar_height
        
        bar_color = (255, 50, 50)
        if getattr(shot, 'shake_timer', 0) > 0 and (shot.shake_timer // 2) % 2 == 0:
             bar_color = (255, 200, 200) 

        pg.draw.rect(screen, (40, 40, 40), (bar_x, bar_y_start, config.GARBAGE_BAR_WIDTH, bar_max_height))
        pg.draw.rect(screen, bar_color, (bar_x, bar_y_fill, config.GARBAGE_BAR_WIDTH, bar_height))
        pg.draw.rect(screen, (200, 200, 200), (bar_x, bar_y_start, config.GARBAGE_BAR_WIDTH, bar_max_height), 1)

    # [NEW] Tetris Effect
    if getattr(shot, 'tetris_timer', 0) > 0:
        effect_font = pg.font.SysFont('Comic Sans MS', 60, bold=True)
        txt = "TETRIS!"
        
        # 製作中空字體 (Hollow Text)
        # 1. 產生白色底圖 (作為邊框)
        base_surf = effect_font.render(txt, True, (255, 255, 255))
        # 2. 產生黑色內文 (用來挖空)
        inner_surf = effect_font.render(txt, True, (0, 0, 0))
        
        # 3. 建立容器
        w, h = base_surf.get_size()
        outline_surf = pg.Surface((w + 4, h + 4), pg.SRCALPHA)
        
        # 4. 在四個角落繪製白色底圖 (偏移 1 pixel 形成細線)
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for ox, oy in offsets:
            outline_surf.blit(base_surf, (ox + 2, oy + 2))
            
        # 5. 在中心挖空 (使用 SUB 模式扣除 Alpha)
        # inner_surf 是黑色 (0,0,0)，Alpha 隨字體反鋸齒變化
        # SUB 模式: (R,G,B,A) - (0,0,0,A_in) = (R,G,B, A - A_in)
        outline_surf.blit(inner_surf, (2, 2), special_flags=pg.BLEND_RGBA_SUB)
        
        # 6. 設定整體透明度 (淡出效果)
        alpha = int(min(255, shot.tetris_timer * 8))
        outline_surf.set_alpha(alpha)
        
        text_rect = outline_surf.get_rect(center=(offset_x + (config.columns * config.grid) // 2, 
                                               offset_y + (config.rows * config.grid) // 3))
        screen.blit(outline_surf, text_rect)

# --- 設定選單 ---

def settings_menu(screen):
    global SHOW_GHOST
    pg.display.set_caption("Tetris Battle - Settings")
    
    font_title = pg.font.SysFont('Comic Sans MS', 50, bold=True)
    
    btn_w, btn_h = 300, 60
    center_x = config.width // 2 - btn_w // 2
    center_y = config.height // 3
    
    btn_back = Button(center_x, center_y + 160, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    while True:
        status_text = "ON" if SHOW_GHOST else "OFF"
        status_color = (50, 200, 50) if SHOW_GHOST else (200, 50, 50)
        btn_ghost = Button(center_x, center_y, btn_w, btn_h, f"Ghost Piece: {status_text}", "TOGGLE_GHOST", color=status_color)
        
        buttons = [btn_ghost, btn_back]
        
        screen.fill(config.background_color)
        
        title_surf = font_title.render("SETTINGS", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//6))
        screen.blit(title_surf, title_rect)
        
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            
            if btn_ghost.is_clicked(event):
                SHOW_GHOST = not SHOW_GHOST
                
            if btn_back.is_clicked(event):
                return 

        for btn in buttons: btn.draw(screen)
        pg.display.update()

# --- 暫停選單 ---

def pause_menu(screen):
    overlay = pg.Surface((config.width, config.height))
    overlay.set_alpha(150)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    btn_w, btn_h = 220, 60
    center_x = config.width // 2 - btn_w // 2
    center_y = config.height // 2 - 100

    btn_resume = Button(center_x, center_y, btn_w, btn_h, "Resume", "RESUME")
    btn_restart = Button(center_x, center_y + 80, btn_w, btn_h, "Restart", "RESTART", color=(200, 150, 50))
    btn_menu = Button(center_x, center_y + 160, btn_w, btn_h, "Main Menu", "MENU", color=(200, 50, 50))
    
    buttons = [btn_resume, btn_restart, btn_menu]
    
    font_pause = pg.font.SysFont('Comic Sans MS', 50, bold=True)
    text_surf = font_pause.render("PAUSED", True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=(config.width//2, center_y - 60))
    screen.blit(text_surf, text_rect)
    
    pg.display.update()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: return "RESUME"
            for btn in buttons:
                if btn.is_clicked(event): return btn.action_code
        for btn in buttons: btn.draw(screen)
        pg.display.update()

# --- 核心遊戲流程 ---
def run_game(screen, clock, font, mode, ai_mode=None, net_mgr=None):
    """
    核心遊戲迴圈
    參數 ai_mode: 當 mode='PVE' 時，指定 'DQN' 或 'HEURISTIC'
    參數 net_mgr: 當 mode='LAN' 時，傳入 NetworkManager 物件
    """
    # --- P1 Initialization (人類玩家) ---
    shot1 = shots.Shot()
    shot1.tetris_timer = 0
    piece1 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
    next_piece1 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    game_over1 = False

    # --- P2 Initialization (對手：人類 P2 或 AI 或 LAN) ---
    shot2 = None
    piece2 = None
    next_piece2 = None
    game_over2 = False
    winner_name = None # 紀錄獲勝者
    
    # AI 相關變數
    ai_nn = None
    ai_target_move = None
    ai_timer = 0
    ai_think_timer = 0

    counter2 = 0
    key_ticker2 = {pg.K_LEFT: 0, pg.K_DOWN: 0, pg.K_RIGHT: 0}

    if mode in ['PVP', 'PVE', 'LAN']:
        shot2 = shots.Shot()
        shot2.tetris_timer = 0
        piece2 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
        next_piece2 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))

        # 如果是 PVE 且選擇了 DQN 模式，嘗試載入模型
        if mode == 'PVE' and ai_mode == 'DQN' and AI_AVAILABLE:
            try:
                ai_nn = AIPlayerNN(model_path='tetris_dqn_new.pt')
                print("AI: Loaded DQN Model")
            except:
                print("Failed to load AI model, fallback to random")
                ai_nn = AIPlayerNN()
        elif mode == 'PVE' and ai_mode == 'HEURISTIC':
             print("AI: Using Heuristic (Expert) Mode")
        elif mode == 'LAN':
             print("LAN Mode: Waiting for data...")

    # --- 畫面位置計算 ---
    p1_draw_x = config.P1_OFFSET_X
    p2_draw_x = config.P2_OFFSET_X

    if mode == 'SOLO':
        total_width = config.GARBAGE_BAR_WIDTH + (config.columns * config.grid) + config.INFO_PANEL_WIDTH
        p1_draw_x = (config.width - total_width) // 2 + config.GARBAGE_BAR_WIDTH

    running = True
    paused = False

    while running:
        # --- 暫停處理 ---
        if paused:
            action = pause_menu(screen)
            if action == "RESUME": paused = False; clock.tick(); continue
            elif action == "RESTART": return "RESTART"
            elif action == "MENU": return "MENU"

        # --- LAN Data Exchange ---
        if mode == 'LAN' and net_mgr:
            if not net_mgr.connected:
                return "MENU" # Disconnected
            
            # 1. Send Local State
            local_data = {
                'status': shot1.status,
                'color': shot1.color,
                'score': shot1.score,
                'lines': shot1.line_count,
                'piece_x': piece1.x,
                'piece_y': piece1.y,
                'piece_shape': piece1.shape,
                'piece_rot': piece1.rotation,
                'piece_color': piece1.color,
                'next_piece_shape': next_piece1.shape,
                'next_piece_color': next_piece1.color,
                'game_over': game_over1,
                'garbage_sent': 0 # Will be set if attack happens
            }
            
            # 2. Receive Remote State
            remote_data = net_mgr.get_latest_data()
            if remote_data:
                shot2.status = remote_data['status']
                shot2.color = remote_data['color']
                shot2.score = remote_data['score']
                shot2.line_count = remote_data['lines']
                
                # Sync Piece 2
                piece2.x = remote_data['piece_x']
                piece2.y = remote_data['piece_y']
                piece2.shape = remote_data['piece_shape']
                piece2.rotation = remote_data['piece_rot']
                piece2.color = remote_data['piece_color']
                
                # Sync Next Piece 2
                next_piece2.shape = remote_data['next_piece_shape']
                next_piece2.color = remote_data['next_piece_color']
                
                # Sync Game Over
                if remote_data['game_over'] and not game_over2:
                    game_over2 = True
                    
                # Sync Incoming Garbage
                if remote_data.get('garbage_sent', 0) > 0:
                    shot1.pending_garbage += remote_data['garbage_sent']

        # --- 事件處理 ---
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE: paused = True
                
                # P1 Controls
                if not game_over1:
                    if event.key == pg.K_w: Handler.rotate(shot1, piece1)
                    if event.key == pg.K_s: key_ticker1[pg.K_s] = 13; Handler.drop(shot1, piece1)
                    if event.key == pg.K_a: key_ticker1[pg.K_a] = 13; Handler.moveLeft(shot1, piece1)
                    if event.key == pg.K_d: key_ticker1[pg.K_d] = 13; Handler.moveRight(shot1, piece1)
                    if event.key == pg.K_LSHIFT: Handler.instantDrop(shot1, piece1)
                
                # P2 Controls (Human only)
                if mode == 'PVP' and not game_over2:
                    if event.key == pg.K_UP: Handler.rotate(shot2, piece2)
                    if event.key == pg.K_DOWN: key_ticker2[pg.K_DOWN] = 13; Handler.drop(shot2, piece2)
                    if event.key == pg.K_LEFT: key_ticker2[pg.K_LEFT] = 13; Handler.moveLeft(shot2, piece2)
                    if event.key == pg.K_RIGHT: key_ticker2[pg.K_RIGHT] = 13; Handler.moveRight(shot2, piece2)
                    if event.key == pg.K_RSHIFT: Handler.instantDrop(shot2, piece2)

        # --- 按鍵持續按壓處理 (DAS) ---
        keys = pg.key.get_pressed()
        if not game_over1:
            if keys[pg.K_a] and key_ticker1[pg.K_a] == 0: key_ticker1[pg.K_a] = 6; Handler.moveLeft(shot1, piece1)
            if keys[pg.K_d] and key_ticker1[pg.K_d] == 0: key_ticker1[pg.K_d] = 6; Handler.moveRight(shot1, piece1)
            if keys[pg.K_s] and key_ticker1[pg.K_s] == 0: key_ticker1[pg.K_s] = 6; Handler.drop(shot1, piece1)
            for k in key_ticker1:
                if key_ticker1[k] > 0: key_ticker1[k] -= 1

        if mode == 'PVP' and not game_over2:
            if keys[pg.K_LEFT] and key_ticker2[pg.K_LEFT] == 0: key_ticker2[pg.K_LEFT] = 6; Handler.moveLeft(shot2, piece2)
            if keys[pg.K_RIGHT] and key_ticker2[pg.K_RIGHT] == 0: key_ticker2[pg.K_RIGHT] = 6; Handler.moveRight(shot2, piece2)
            if keys[pg.K_DOWN] and key_ticker2[pg.K_DOWN] == 0: key_ticker2[pg.K_DOWN] = 6; Handler.drop(shot2, piece2)
            for k in key_ticker2:
                if key_ticker2[k] > 0: key_ticker2[k] -= 1

        # --- AI Logic (PVE Mode) ---
        if mode == 'PVE' and not game_over2:
            if ai_target_move is None:
                # 思考延遲模擬
                if ai_think_timer < AI_THINKING_DELAY:
                    ai_think_timer += 1
                else:
                    # === 決定使用哪個大腦 ===
                    if ai_mode == 'DQN' and ai_nn:
                        # 使用神經網路
                        ai_target_move = ai_nn.find_best_move(copy.deepcopy(shot2), copy.deepcopy(piece2), copy.deepcopy(next_piece2))
                    elif ai_mode == 'HEURISTIC':
                        # 使用啟發式演算法 (from main_2p.py)
                        ai_target_move = get_ai_move_heuristic(shot2, piece2)
                    else:
                        # 無 AI 時的隨機移動
                        ai_target_move = (random.randint(0, config.columns-3), random.randint(0, 3))

                    # 若 AI 放棄思考 (None)，保持原位
                    if ai_target_move is None: 
                        ai_target_move = (piece2.x, piece2.rotation)
                    
                    ai_think_timer = 0
            else:
                # 執行移動 (逐步移動到目標位置)
                target_x, target_r = ai_target_move
                aligned = (piece2.x == target_x) and (piece2.rotation == target_r)
                
                if ai_timer >= AI_MOVE_DELAY:
                    ai_timer = 0
                    if piece2.rotation != target_r:
                        Handler.rotate(shot2, piece2)
                    elif piece2.x < target_x:
                        Handler.moveRight(shot2, piece2)
                    elif piece2.x > target_x:
                        Handler.moveLeft(shot2, piece2)
                else:
                    ai_timer += 1
                
                # 自動下落
                drop_threshold = config.difficulty
                if aligned: drop_threshold = max(2, config.difficulty // 8) # 到位後加速下落
                
                if counter2 >= drop_threshold:
                    Handler.drop(shot2, piece2)
                    counter2 = 0
                else:
                    counter2 += 1

        # --- Gravity (自動下落) ---
        if not game_over1:
            if counter1 >= config.difficulty: Handler.drop(shot1, piece1); counter1 = 0
            else: counter1 += 1

        if mode == 'PVP' and not game_over2:
            if counter2 >= config.difficulty: Handler.drop(shot2, piece2); counter2 = 0
            else: counter2 += 1

        # --- Game Logic: P1 Check ---
        if not game_over1 and piece1.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot1, piece1)
            if clears == 4: shot1.tetris_timer = 60
            atk1 = 0
            if mode != 'SOLO':
                is_power_move = (clears == 4)
                shot1.combo_count = shot1.combo_count + 1 if clears > 0 else 0
                atk1 = Handler.calculateAttack(clears, shot1.combo_count, shot1.is_b2b, all_clear)
                shot1.is_b2b = is_power_move if is_power_move else (False if clears > 0 else shot1.is_b2b)

                if atk1 > 0 and shot1.pending_garbage > 0:
                    cancel = min(atk1, shot1.pending_garbage)
                    shot1.pending_garbage -= cancel
                    atk1 -= cancel

            if clears == 0 and shot1.pending_garbage > 0:
                Handler.insertGarbage(shot1, shot1.pending_garbage)
                shot1.pending_garbage = 0
                shot1.shake_timer = 20
            
            if mode != 'SOLO' and atk1 > 0:
                if mode == 'LAN' and net_mgr:
                    local_data['garbage_sent'] = atk1 # Mark for sending
                else:
                    shot2.pending_garbage += atk1

            piece1 = next_piece1
            next_piece1 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))

            if Handler.isDefeat(shot1, piece1): 
                game_over1 = True
                if mode != 'SOLO' and winner_name is None:
                    winner_name = "AI Wins!" if mode == 'PVE' else "Player 2 Wins!"

        # --- Game Logic: P2 Check ---
        if mode != 'SOLO' and mode != 'LAN' and not game_over2 and piece2.is_fixed:
            # 方塊落地後，重置 AI 思考
            if mode == 'PVE': ai_target_move = None; ai_think_timer = 0
            
            clears, all_clear = Handler.eliminateFilledRows(shot2, piece2)
            if clears == 4: shot2.tetris_timer = 60
            atk2 = 0
            
            is_power_move = (clears == 4)
            shot2.combo_count = shot2.combo_count + 1 if clears > 0 else 0
            atk2 = Handler.calculateAttack(clears, shot2.combo_count, shot2.is_b2b, all_clear)
            shot2.is_b2b = is_power_move if is_power_move else (False if clears > 0 else shot2.is_b2b)

            if atk2 > 0 and shot2.pending_garbage > 0:
                cancel = min(atk2, shot2.pending_garbage)
                shot2.pending_garbage -= cancel
                atk2 -= cancel

            if clears == 0 and shot2.pending_garbage > 0:
                Handler.insertGarbage(shot2, shot2.pending_garbage)
                shot2.pending_garbage = 0
                shot2.shake_timer = 20

            if atk2 > 0:
                shot1.pending_garbage += atk2

            piece2 = next_piece2
            next_piece2 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))

            if Handler.isDefeat(shot2, piece2): 
                game_over2 = True
                if mode != 'SOLO' and winner_name is None:
                    winner_name = "Player 1 Wins!"

        # --- Send Data (LAN) ---
        if mode == 'LAN' and net_mgr:
            net_mgr.send(local_data)

        # --- 勝負判定 ---
        if mode == 'SOLO':
            if game_over1:
                return "GAME_OVER", {"winner": "Solo Finish", "score": shot1.score, "lines": shot1.line_count}
        else:
            # 修改: 必須兩者都 Game Over 才結束
            if game_over1 and game_over2:
                final_winner = winner_name if winner_name else "Draw"
                # 顯示獲勝者的分數 (或是 P1 的分數，視需求而定)
                winning_score = shot1.score
                if "Player 2" in final_winner or "AI" in final_winner:
                    winning_score = shot2.score
                
                return "GAME_OVER", {"winner": final_winner, "score": winning_score, "lines": shot1.line_count}

        # --- 畫面更新 ---
        if getattr(shot1, 'tetris_timer', 0) > 0: shot1.tetris_timer -= 1
        if shot2 and getattr(shot2, 'tetris_timer', 0) > 0: shot2.tetris_timer -= 1
        
        screen.fill(config.background_color)
        
        # 繪製 P1
        draw_player_ui(screen, shot1, piece1, next_piece1, font, p1_draw_x, None, None, None, None, "Player 1")
        if game_over1:
            # 繪製 Game Over 遮罩
            s = pg.Surface((config.columns * config.grid, config.rows * config.grid))
            s.set_alpha(150)
            s.fill((0,0,0))
            screen.blit(s, (p1_draw_x, 0))
            text = font.render("GAME OVER", True, (255, 50, 50))
            text_rect = text.get_rect(center=(p1_draw_x + (config.columns * config.grid)//2, (config.rows * config.grid)//2))
            screen.blit(text, text_rect)
        
        # 繪製 P2
        if mode != 'SOLO':
            # 根據模式顯示名字
            p2_name = "Player 2"
            if mode == 'PVE':
                p2_name = "AI (DQN)" if ai_mode == 'DQN' else "AI (Heuristic)"
            elif mode == 'LAN':
                p2_name = "Network Opponent"
                
            draw_player_ui(screen, shot2, piece2, next_piece2, font, p2_draw_x, None, None, None, None, p2_name)
            if game_over2:
                # 繪製 Game Over 遮罩
                s = pg.Surface((config.columns * config.grid, config.rows * config.grid))
                s.set_alpha(150)
                s.fill((0,0,0))
                screen.blit(s, (p2_draw_x, 0))
                text = font.render("GAME OVER", True, (255, 50, 50))
                text_rect = text.get_rect(center=(p2_draw_x + (config.columns * config.grid)//2, (config.rows * config.grid)//2))
                screen.blit(text, text_rect)

        pg.display.update()
        clock.tick(config.fps)

# --- 主選單與結算 ---

def main_menu(screen, font):
    pg.display.set_caption("Tetris Battle - Menu")
    btn_w, btn_h = 200, 60
    center_x = config.width // 2 - btn_w // 2
    start_y = config.height // 4
    
    btn_solo = Button(center_x, start_y, btn_w, btn_h, "Solo Mode", "SOLO")
    btn_pvp = Button(center_x, start_y + 80, btn_w, btn_h, "1v1 Local", "PVP", color=(50, 100, 200))
    btn_pve = Button(center_x, start_y + 160, btn_w, btn_h, "1vAI Battle", "PVE", color=(200, 50, 50))
    btn_lan = Button(center_x, start_y + 240, btn_w, btn_h, "LAN Battle", "LAN", color=(150, 50, 150))
    btn_settings = Button(center_x, start_y + 320, btn_w, btn_h, "Settings", "SETTINGS", color=(100, 100, 100))
    btn_exit = Button(center_x, start_y + 400, btn_w, btn_h, "Exit Game", "EXIT", color=(50, 50, 50))
    
    buttons = [btn_solo, btn_pvp, btn_pve, btn_lan, btn_settings, btn_exit]

    while True:
        screen.fill(config.background_color)
        title_surf = pg.font.SysFont('Comic Sans MS', 60, bold=True).render("TETRIS BATTLE", True, (255, 215, 0))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//8))
        screen.blit(title_surf, title_rect)
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            for btn in buttons:
                if btn.is_clicked(event): return btn.action_code
        for btn in buttons: btn.draw(screen)
        pg.display.update()

# --- [新增] AI 選擇選單 ---
def ai_selection_menu(screen, font):
    pg.display.set_caption("Select AI Opponent")
    
    btn_w, btn_h = 320, 60
    center_x = config.width // 2 - btn_w // 2
    start_y = config.height // 3
    
    # 定義三個按鈕
    btn_dqn = Button(center_x, start_y, btn_w, btn_h, "DQN AI (Neural Net)", "DQN", color=(50, 100, 200))
    btn_heuristic = Button(center_x, start_y + 80, btn_w, btn_h, "Heuristic AI (Expert)", "HEURISTIC", color=(200, 50, 50))
    btn_back = Button(center_x, start_y + 200, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    buttons = [btn_dqn, btn_heuristic, btn_back]
    
    font_title = pg.font.SysFont('Comic Sans MS', 40, bold=True)
    
    while True:
        screen.fill(config.background_color)
        
        title_surf = font_title.render("CHOOSE YOUR OPPONENT", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//6))
        screen.blit(title_surf, title_rect)
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit(); sys.exit()
            
            # 檢查按鈕點擊
            for btn in buttons:
                if btn.is_clicked(event):
                    return btn.action_code # 回傳 "DQN", "HEURISTIC", 或 "BACK"
                    
        for btn in buttons:
            btn.draw(screen)
            
        pg.display.update()


# --- LAN Menu ---
def lan_menu(screen, font):
    pg.display.set_caption("LAN Battle Setup")
    
    btn_w, btn_h = 300, 60
    center_x = config.width // 2 - btn_w // 2
    start_y = config.height // 3
    
    btn_host = Button(center_x, start_y, btn_w, btn_h, "Host Game", "HOST", color=(50, 150, 50))
    btn_join = Button(center_x, start_y + 80, btn_w, btn_h, "Join Game", "JOIN", color=(50, 100, 200))
    btn_back = Button(center_x, start_y + 200, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    buttons = [btn_host, btn_join, btn_back]
    
    # Input box for IP
    ip_text = "127.0.0.1"
    input_active = False
    input_rect = pg.Rect(center_x, start_y + 150, btn_w, 40)
    
    net_mgr = network_utils.NetworkManager()
    
    while True:
        screen.fill(config.background_color)
        
        title_surf = pg.font.SysFont('Comic Sans MS', 40, bold=True).render("LAN SETUP", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//6))
        screen.blit(title_surf, title_rect)
        
        # Draw Input Box
        color = (255, 255, 255) if input_active else (150, 150, 150)
        pg.draw.rect(screen, color, input_rect, 2)
        text_surface = font.render(ip_text, True, (255, 255, 255))
        
        # Center text vertically
        text_rect = text_surface.get_rect(midleft=(input_rect.x + 10, input_rect.centery))
        screen.blit(text_surface, text_rect)
        
        # Label for Input
        label_surf = font.render("Host IP:", True, (200, 200, 200))
        label_rect = label_surf.get_rect(midright=(input_rect.x - 10, input_rect.centery))
        screen.blit(label_surf, label_rect)
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit(); sys.exit()
            
            if event.type == pg.MOUSEBUTTONDOWN:
                if input_rect.collidepoint(event.pos):
                    input_active = True
                else:
                    input_active = False
                
                for btn in buttons:
                    if btn.is_clicked(event):
                        if btn.action_code == "BACK":
                            return None, None
                        elif btn.action_code == "HOST":
                            # Show waiting screen with non-blocking loop
                            local_ip = net_mgr.get_local_ip()

                            # Start hosting in a separate thread
                            host_thread = threading.Thread(target=net_mgr.host_game, daemon=True)
                            host_thread.start()
                            
                            waiting = True
                            clock = pg.time.Clock()
                            
                            while waiting:
                                # 1. Check connection success
                                if net_mgr.connected:
                                    return "LAN", net_mgr
                                
                                # 2. Check if thread died (error)
                                if not host_thread.is_alive():
                                    waiting = False
                                
                                # 3. Handle Events
                                for e in pg.event.get():
                                    if e.type == pg.QUIT:
                                        net_mgr.close()
                                        pg.quit(); sys.exit()
                                    if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                                        net_mgr.close()
                                        waiting = False
                                        # Re-create manager for next attempt
                                        net_mgr = network_utils.NetworkManager()
                                
                                # 4. Draw Waiting Screen
                                screen.fill(config.background_color)
                                
                                txt1 = font.render("Waiting for connection...", True, (255, 255, 255))
                                txt2 = font.render(f"Your IP: {local_ip}", True, (255, 215, 0))
                                txt3 = font.render("Press ESC to Cancel", True, (150, 150, 150))
                                
                                r1 = txt1.get_rect(center=(config.width//2, config.height//2 - 50))
                                r2 = txt2.get_rect(center=(config.width//2, config.height//2 + 20))
                                r3 = txt3.get_rect(center=(config.width//2, config.height//2 + 100))
                                
                                screen.blit(txt1, r1)
                                screen.blit(txt2, r2)
                                screen.blit(txt3, r3)
                                
                                pg.display.update()
                                clock.tick(30)
                            
                        elif btn.action_code == "JOIN":
                            # Start joining in a separate thread
                            join_thread = threading.Thread(target=net_mgr.join_game, args=(ip_text,), daemon=True)
                            join_thread.start()
                            
                            waiting = True
                            clock = pg.time.Clock()
                            
                            while waiting:
                                # 1. Check connection success
                                if net_mgr.connected:
                                    return "LAN", net_mgr
                                
                                # 2. Check if thread finished (failed)
                                if not join_thread.is_alive():
                                    waiting = False
                                    # Show error
                                    err_start = pg.time.get_ticks()
                                    while pg.time.get_ticks() - err_start < 3000:
                                        screen.fill(config.background_color)
                                        err_surf = font.render("Connection Failed!", True, (255, 50, 50))
                                        hint_surf = pg.font.SysFont('Arial', 20).render("Check IP or Firewall settings", True, (200, 200, 200))
                                        
                                        screen.blit(err_surf, err_surf.get_rect(center=(config.width//2, config.height//2 - 20)))
                                        screen.blit(hint_surf, hint_surf.get_rect(center=(config.width//2, config.height//2 + 30)))
                                        
                                        pg.display.update()
                                        pg.event.pump()
                                    
                                    # Reset manager for next try
                                    net_mgr = network_utils.NetworkManager()
                                
                                # 3. Handle Events
                                for e in pg.event.get():
                                    if e.type == pg.QUIT:
                                        net_mgr.close()
                                        pg.quit(); sys.exit()
                                    if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                                        net_mgr.close()
                                        waiting = False
                                        net_mgr = network_utils.NetworkManager()
                                
                                # 4. Draw Waiting Screen
                                screen.fill(config.background_color)
                                txt1 = font.render(f"Connecting to {ip_text}...", True, (255, 255, 255))
                                txt2 = font.render("Press ESC to Cancel", True, (150, 150, 150))
                                
                                r1 = txt1.get_rect(center=(config.width//2, config.height//2))
                                r2 = txt2.get_rect(center=(config.width//2, config.height//2 + 60))
                                
                                screen.blit(txt1, r1)
                                screen.blit(txt2, r2)
                                
                                pg.display.update()
                                clock.tick(30)
            
            if event.type == pg.KEYDOWN:
                if input_active:
                    if event.key == pg.K_RETURN:
                        input_active = False
                    elif event.key == pg.K_BACKSPACE:
                        ip_text = ip_text[:-1]
                    else:
                        ip_text += event.unicode
                        
        for btn in buttons:
            btn.draw(screen)
            
        pg.display.update()

def game_over_screen(screen, result_data):
    pg.display.set_caption("Game Over")
    font_large = pg.font.SysFont('Comic Sans MS', 50, bold=True)
    font_small = pg.font.SysFont('Arial', 30)
    
    btn_restart = Button(config.width//2 - 100, config.height//2 + 80, 200, 60, "Play Again", "RESTART")
    btn_menu = Button(config.width//2 - 100, config.height//2 + 160, 200, 60, "Main Menu", "MENU", color=(150, 50, 50))
    buttons = [btn_restart, btn_menu]

    while True:
        s = pg.Surface((config.width, config.height))
        s.set_alpha(10)
        s.fill((0,0,0))
        screen.blit(s, (0,0))
        
        win_text = result_data.get("winner", "Game Over")
        win_surf = font_large.render(win_text, True, (255, 50, 50))
        win_rect = win_surf.get_rect(center=(config.width//2, config.height//4))
        
        score_text = f"Your Score: {result_data.get('score', 0)}"
        lines_text = f"Lines Cleared: {result_data.get('lines', 0)}"
        
        score_surf = font_small.render(score_text, True, (255, 255, 255))
        lines_surf = font_small.render(lines_text, True, (255, 255, 255))
        
        score_rect = score_surf.get_rect(center=(config.width//2, config.height//4 + 60))
        lines_rect = lines_surf.get_rect(center=(config.width//2, config.height//4 + 100))
        
        bg_rect = pg.Rect(0, 0, 400, 250)
        bg_rect.center = (config.width//2, config.height//4 + 50)
        pg.draw.rect(screen, (30, 30, 30), bg_rect, border_radius=10)
        pg.draw.rect(screen, (255, 255, 255), bg_rect, 2, border_radius=10)

        screen.blit(win_surf, win_rect)
        screen.blit(score_surf, score_rect)
        screen.blit(lines_surf, lines_rect)
        
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            for btn in buttons:
                if btn.is_clicked(event): return btn.action_code
        
        for btn in buttons: btn.draw(screen)
        pg.display.update()

def main():
    pg.init()
    pg.font.init()
    
    screen = pg.display.set_mode((config.width, config.height))
    clock = pg.time.Clock()
    font = pg.font.SysFont(*config.font)
    
    current_mode = None
    
    while True:
        # 1. 顯示主選單
        choice = main_menu(screen, font)
        
        if choice == "EXIT":
            pg.quit()
            sys.exit()
        elif choice == "SETTINGS":
            settings_menu(screen)
            continue
        
        # 2. 處理 AI 選擇邏輯
        selected_ai_mode = None # 預設無
        net_mgr = None # 預設無
        
        if choice == "PVE":
            # 如果選了 PVE，先跳出選擇 AI 難度的視窗
            ai_choice = ai_selection_menu(screen, font)
            if ai_choice == "BACK":
                continue # 放棄，回到主選單
            selected_ai_mode = ai_choice # 紀錄是 DQN 還是 HEURISTIC
            
        elif choice == "LAN":
            # 如果選了 LAN，跳出連線選單
            lan_mode, mgr = lan_menu(screen, font)
            if lan_mode is None:
                continue
            net_mgr = mgr
        
        current_mode = choice
        
        # 3. 進入遊戲
        while True:
            # 將 ai_mode 傳入 run_game
            result = run_game(screen, clock, font, current_mode, ai_mode=selected_ai_mode, net_mgr=net_mgr)
            
            if result == "MENU":
                if net_mgr: net_mgr.close()
                break # 回到主選單
            elif result == "RESTART":
                # LAN 模式下 Restart 比較複雜，這裡先簡單處理：斷線重連
                # 實際上應該發送 Restart 訊號，但為了簡化，LAN 模式下 Restart 回到選單
                if current_mode == 'LAN':
                    if net_mgr: net_mgr.close()
                    break
                continue # 重新開始這一局 (保持同樣的 AI 設定)
            
            if isinstance(result, tuple) and result[0] == "GAME_OVER":
                action = game_over_screen(screen, result[1])
                if action == "RESTART":
                    if current_mode == 'LAN':
                        if net_mgr: net_mgr.close()
                        break
                    continue
                elif action == "MENU":
                    if net_mgr: net_mgr.close()
                    break

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()