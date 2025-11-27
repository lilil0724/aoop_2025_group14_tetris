import pygame as pg
import sys
import copy
import random

# 引用模組
import config
import pieces
import shots
import Handler

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
        pg.draw.rect(screen, current_color, self.rect, border_radius=10)
        pg.draw.rect(screen, (255, 255, 255), self.rect, 3, border_radius=10)
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
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
            draw_color = (0, 0, 0) if shot.status[y][x] == 0 else color
            if shot.status[y][x] == 0: draw_color = config.background_color
            
            pg.draw.rect(screen, draw_color, (
                offset_x + x * config.grid,
                offset_y + y * config.grid,
                config.grid,
                config.grid
            ))

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
                pg.draw.rect(screen, piece.color, (
                    offset_x + x * config.grid,
                    offset_y + y * config.grid,
                    config.grid,
                    config.grid
                ))

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
        pg.draw.rect(screen, next_piece.color, (next_center_x + x * config.grid, next_center_y + y * config.grid, config.grid, config.grid))

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

def run_game(screen, clock, font, mode):
    
    # P1 Initialization
    shot1 = shots.Shot()
    piece1 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
    next_piece1 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    game_over1 = False
    
    # P2 Initialization
    shot2 = None
    piece2 = None
    next_piece2 = None
    game_over2 = False
    
    ai_nn = None
    ai_target_move = None 
    ai_timer = 0           
    ai_think_timer = 0     
    counter2 = 0           
    key_ticker2 = {pg.K_LEFT: 0, pg.K_DOWN: 0, pg.K_RIGHT: 0} 
    
    if mode in ['PVP', 'PVE']:
        shot2 = shots.Shot()
        piece2 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
        next_piece2 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
        if mode == 'PVE' and AI_AVAILABLE:
            try:
                ai_nn = AIPlayerNN(model_path='tetris_dqn_new.pt')
            except:
                print("Failed to load AI model")
                ai_nn = AIPlayerNN() 

    # SOLO 模式置中
    p1_draw_x = config.P1_OFFSET_X
    p2_draw_x = config.P2_OFFSET_X
    if mode == 'SOLO':
        total_width = config.GARBAGE_BAR_WIDTH + (config.columns * config.grid) + config.INFO_PANEL_WIDTH
        p1_draw_x = (config.width - total_width) // 2 + config.GARBAGE_BAR_WIDTH

    running = True
    paused = False

    while running:
        if paused:
            action = pause_menu(screen)
            if action == "RESUME": paused = False; clock.tick(); continue
            elif action == "RESTART": return "RESTART"
            elif action == "MENU": return "MENU"

        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE: paused = True 
                if not game_over1:
                    if event.key == pg.K_w: Handler.rotate(shot1, piece1)
                    if event.key == pg.K_s: key_ticker1[pg.K_s] = 13; Handler.drop(shot1, piece1)
                    if event.key == pg.K_a: key_ticker1[pg.K_a] = 13; Handler.moveLeft(shot1, piece1)
                    if event.key == pg.K_d: key_ticker1[pg.K_d] = 13; Handler.moveRight(shot1, piece1)
                    if event.key == pg.K_LSHIFT: Handler.instantDrop(shot1, piece1)

                if mode == 'PVP' and not game_over2:
                    if event.key == pg.K_UP: Handler.rotate(shot2, piece2)
                    if event.key == pg.K_DOWN: key_ticker2[pg.K_DOWN] = 13; Handler.drop(shot2, piece2)
                    if event.key == pg.K_LEFT: key_ticker2[pg.K_LEFT] = 13; Handler.moveLeft(shot2, piece2)
                    if event.key == pg.K_RIGHT: key_ticker2[pg.K_RIGHT] = 13; Handler.moveRight(shot2, piece2)
                    if event.key == pg.K_RSHIFT: Handler.instantDrop(shot2, piece2)

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

        if mode == 'PVE' and not game_over2:
            if ai_target_move is None:
                if ai_think_timer < AI_THINKING_DELAY: ai_think_timer += 1
                else:
                    if ai_nn:
                        ai_target_move = ai_nn.find_best_move(copy.deepcopy(shot2), copy.deepcopy(piece2), copy.deepcopy(next_piece2))
                        if ai_target_move is None: ai_target_move = (piece2.x, piece2.rotation)
                    else:
                        ai_target_move = (random.randint(0, config.columns-3), random.randint(0, 3))
                    ai_think_timer = 0
            else:
                target_x, target_r = ai_target_move
                aligned = (piece2.x == target_x) and (piece2.rotation == target_r)
                if ai_timer >= AI_MOVE_DELAY:
                    ai_timer = 0
                    if piece2.rotation != target_r: Handler.rotate(shot2, piece2)
                    elif piece2.x < target_x: Handler.moveRight(shot2, piece2)
                    elif piece2.x > target_x: Handler.moveLeft(shot2, piece2)
                else: ai_timer += 1
                drop_threshold = config.difficulty
                if aligned: drop_threshold = max(2, config.difficulty // 8) 
                if counter2 >= drop_threshold: Handler.drop(shot2, piece2); counter2 = 0
                else: counter2 += 1
        
        if not game_over1:
            if counter1 >= config.difficulty: Handler.drop(shot1, piece1); counter1 = 0
            else: counter1 += 1
        if mode == 'PVP' and not game_over2:
            if counter2 >= config.difficulty: Handler.drop(shot2, piece2); counter2 = 0
            else: counter2 += 1

        # P1 Check
        if not game_over1 and piece1.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot1, piece1)
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
                shot2.pending_garbage += atk1
            piece1 = next_piece1
            next_piece1 = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
            if Handler.isDefeat(shot1, piece1): game_over1 = True

        # P2 Check
        if mode != 'SOLO' and not game_over2 and piece2.is_fixed:
            if mode == 'PVE': ai_target_move = None; ai_think_timer = 0
            clears, all_clear = Handler.eliminateFilledRows(shot2, piece2)
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
            if Handler.isDefeat(shot2, piece2): game_over2 = True

        # 遊戲結束
        if mode == 'SOLO':
            if game_over1: 
                return "GAME_OVER", {"winner": "Solo Finish", "score": shot1.score, "lines": shot1.line_count}
        else:
            if game_over1: 
                return "GAME_OVER", {"winner": "Player 2 Wins!", "score": shot1.score, "lines": shot1.line_count}
            if game_over2: 
                return "GAME_OVER", {"winner": "Player 1 Wins!", "score": shot1.score, "lines": shot1.line_count}

        screen.fill(config.background_color)
        draw_player_ui(screen, shot1, piece1, next_piece1, font, p1_draw_x, None, None, None, None, "Player 1")
        if mode != 'SOLO':
            p2_name = "AI Bot" if mode == 'PVE' else "Player 2"
            draw_player_ui(screen, shot2, piece2, next_piece2, font, p2_draw_x, None, None, None, None, p2_name)
        
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
    btn_settings = Button(center_x, start_y + 240, btn_w, btn_h, "Settings", "SETTINGS", color=(100, 100, 100))
    btn_exit = Button(center_x, start_y + 320, btn_w, btn_h, "Exit Game", "EXIT", color=(50, 50, 50))
    
    buttons = [btn_solo, btn_pvp, btn_pve, btn_settings, btn_exit]

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
        choice = main_menu(screen, font)
        
        if choice == "EXIT":
            pg.quit()
            sys.exit()
        elif choice == "SETTINGS":
            settings_menu(screen)
            continue
        
        current_mode = choice
        
        while True:
            result = run_game(screen, clock, font, current_mode)
            if result == "MENU": break 
            elif result == "RESTART": continue 
            
            if isinstance(result, tuple) and result[0] == "GAME_OVER":
                action = game_over_screen(screen, result[1])
                if action == "RESTART": continue
                elif action == "MENU": break

if __name__ == "__main__":
    main()