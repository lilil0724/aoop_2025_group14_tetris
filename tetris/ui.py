import pygame as pg
import config
import Handler
import copy
import random
import settings

# --- UI 元件 (Button 保持不變) ---
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
        light = (min(255, int(r * 1.4)), min(255, int(g * 1.4)), min(255, int(b * 1.4)))
        dark = (int(r * 0.6), int(g * 0.6), int(b * 0.6))
        bevel = 6
        pg.draw.rect(screen, current_color, (x + bevel, y + bevel, w - 2*bevel, h - 2*bevel))
        pg.draw.polygon(screen, light, [(x, y), (x + w, y), (x + w - bevel, y + bevel), (x + bevel, y + bevel)])
        pg.draw.polygon(screen, light, [(x, y), (x + bevel, y + bevel), (x + bevel, y + h - bevel), (x, y + h)])
        pg.draw.polygon(screen, dark, [(x, y + h), (x + w, y + h), (x + w - bevel, y + h - bevel), (x + bevel, y + h - bevel)])
        pg.draw.polygon(screen, dark, [(x + w, y), (x + w, y + h), (x + w - bevel, y + h - bevel), (x + w - bevel, y + bevel)])
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        if self.rect.collidepoint(mouse_pos) and pg.mouse.get_pressed()[0]:
             text_rect.move_ip(2, 2)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, color=(200, 200, 200), handle_color=(50, 150, 250)):
        self.rect = pg.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.color = color
        self.handle_color = handle_color
        self.dragging = False
        
        # Handle (Knob)
        self.handle_w = h + 10
        self.handle_h = h + 10
        self.handle_rect = pg.Rect(0, 0, self.handle_w, self.handle_h)
        self.update_handle_pos()

    def update_handle_pos(self):
        # Map value to x position
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        center_x = self.rect.x + ratio * self.rect.w
        self.handle_rect.center = (center_x, self.rect.centery)

    def update_value_from_pos(self, x):
        # Map x position to value
        # Clamp x
        x = max(self.rect.x, min(x, self.rect.right))
        ratio = (x - self.rect.x) / self.rect.w
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        self.update_handle_pos()

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                self.update_value_from_pos(event.pos[0])
                return True # Value changed
        
        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
            
        elif event.type == pg.MOUSEMOTION:
            if self.dragging:
                self.update_value_from_pos(event.pos[0])
                return True # Value changed
                
        return False

    def draw(self, screen):
        # Draw Track
        pg.draw.rect(screen, self.color, self.rect, border_radius=self.rect.height//2)
        
        # Draw Filled Track (Left of handle)
        fill_rect = pg.Rect(self.rect.x, self.rect.y, self.handle_rect.centerx - self.rect.x, self.rect.height)
        if fill_rect.width > 0:
            pg.draw.rect(screen, self.handle_color, fill_rect, border_radius=self.rect.height//2)

        # Draw Handle
        pg.draw.rect(screen, (255, 255, 255), self.handle_rect, border_radius=self.handle_h//2)
        pg.draw.rect(screen, self.handle_color, self.handle_rect, 2, border_radius=self.handle_h//2)

# --- 輔助功能: 投影位置計算 ---
def get_ghost_piece(shot, piece):
    ghost = copy.deepcopy(piece)
    while True:
        can_move_down = True
        for y, x in Handler.getCellsAbsolutePosition(ghost):
            ny = y + 1
            if ny >= config.rows:
                can_move_down = False; break
            if ny >= 0 and shot.status[ny][x] == 2:
                can_move_down = False; break
        if can_move_down: ghost.y += 1
        else: break
    return ghost

def draw_3d_block(surface, color, x, y, size):
    r, g, b = color
    light = (min(255, int(r * 1.4)), min(255, int(g * 1.4)), min(255, int(b * 1.4)))
    dark = (int(r * 0.6), int(g * 0.6), int(b * 0.6))
    bevel = size // 6
    pg.draw.rect(surface, color, (x + bevel, y + bevel, size - 2*bevel, size - 2*bevel))
    pg.draw.polygon(surface, light, [(x, y), (x + size, y), (x + size - bevel, y + bevel), (x + bevel, y + bevel)])
    pg.draw.polygon(surface, light, [(x, y), (x + bevel, y + bevel), (x + bevel, y + size - bevel), (x, y + size)])
    pg.draw.polygon(surface, dark, [(x, y + size), (x + size, y + size), (x + size - bevel, y + size - bevel), (x + bevel, y + size - bevel)])
    pg.draw.polygon(surface, dark, [(x + size, y), (x + size, y + size), (x + size - bevel, y + size - bevel), (x + size - bevel, y + bevel)])

def _draw_effect_text(surface, text, color, timer, draw_x, draw_y, offset_y=0):
    """ 輔助函式: 繪製特效文字 (含描邊與淡出) """
    effect_font = pg.font.SysFont('Comic Sans MS', 40, bold=True)
    base_surf = effect_font.render(text, True, color)
    inner_surf = effect_font.render(text, True, (0, 0, 0))
    w_t, h_t = base_surf.get_size()
    
    # 描邊效果
    outline_surf = pg.Surface((w_t + 4, h_t + 4), pg.SRCALPHA)
    offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for ox, oy in offsets:
        outline_surf.blit(base_surf, (ox + 2, oy + 2))
    outline_surf.blit(inner_surf, (2, 2), special_flags=pg.BLEND_RGBA_SUB)
    
    # 淡出效果 (timer 剩餘越少越透明)
    alpha = int(min(255, timer * 8))
    outline_surf.set_alpha(alpha)
    
    center_x = draw_x + (config.columns * config.grid) // 2
    center_y = draw_y + (config.rows * config.grid) // 3 + offset_y
    text_rect = outline_surf.get_rect(center=(center_x, center_y))
    surface.blit(outline_surf, text_rect)

def draw_player_ui_surface(shot, piece, next_piece, font, player_name="Player"):
    w = config.GARBAGE_BAR_WIDTH + (config.columns * config.grid) + 150
    h = config.rows * config.grid + 50
    surface = pg.Surface((w, h), pg.SRCALPHA)
    
    off_x = config.GARBAGE_BAR_WIDTH
    off_y = 30
    
    # 震動效果
    shake_x, shake_y = 0, 0
    if getattr(shot, 'shake_timer', 0) > 0:
        shake_x = random.randint(-4, 4)
        shake_y = random.randint(-4, 4)
        shot.shake_timer -= 1
        border_rect = pg.Rect(off_x - 2 + shake_x, off_y - 2 + shake_y, config.columns * config.grid + 4, config.rows * config.grid + 4)
        pg.draw.rect(surface, (255, 50, 50), border_rect, 4)

    draw_x = off_x + shake_x
    draw_y = off_y + shake_y

    # 1. 繪製盤面
    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            if shot.status[y][x] != 0:
                draw_3d_block(surface, color, draw_x + x * config.grid, draw_y + y * config.grid, config.grid)

    # 2. 繪製 Ghost
    if settings.SHOW_GHOST and not piece.is_fixed:
        ghost = get_ghost_piece(shot, piece)
        for y, x in Handler.getCellsAbsolutePosition(ghost):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                pg.draw.rect(surface, (180, 180, 180), (draw_x + x * config.grid, draw_y + y * config.grid, config.grid, config.grid), 2) 

    # 3. 繪製實體方塊
    if not piece.is_fixed:
        for y, x in Handler.getCellsAbsolutePosition(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                draw_3d_block(surface, piece.color, draw_x + x * config.grid, draw_y + y * config.grid, config.grid)

    # 4. Grid 與 外框
    for i in range(config.columns + 1):
        pg.draw.line(surface, (80, 80, 80), (draw_x + i * config.grid, draw_y), (draw_x + i * config.grid, draw_y + config.rows * config.grid), 1)
    for i in range(config.rows + 1):
        pg.draw.line(surface, (80, 80, 80), (draw_x, draw_y + i * config.grid), (draw_x + config.columns * config.grid, draw_y + i * config.grid), 1)
    pg.draw.rect(surface, (255, 255, 255), (draw_x, draw_y, config.columns * config.grid, config.rows * config.grid), 3)
    
    # --- UI 資訊區 ---
    info_start_x = draw_x + (config.columns * config.grid) + 10
    label_font = pg.font.SysFont('Arial', 20, bold=True)
    name_surf = label_font.render(player_name, True, (200, 200, 200))
    name_rect = name_surf.get_rect(center=(draw_x + (config.columns * config.grid)//2, 15))
    surface.blit(name_surf, name_rect)

    # Next Piece
    next_label_font = pg.font.SysFont('Arial', 18)
    surface.blit(next_label_font.render("Next:", True, (150, 150, 150)), (info_start_x, draw_y + 10))
    next_center_x = info_start_x + 40
    next_center_y = draw_y + 60
    for y, x in next_piece.getCells():
        draw_3d_block(surface, next_piece.color, next_center_x + x * config.grid, next_center_y + y * config.grid, config.grid)

    # Score & Lines
    info_y_start = draw_y + 200
    score_font = pg.font.SysFont('Arial', 18)
    surface.blit(score_font.render(f'Score:', True, (150, 150, 150)), (info_start_x, info_y_start))
    surface.blit(score_font.render(f'{shot.score}', True, (255, 255, 255)), (info_start_x, info_y_start + 20))
    surface.blit(score_font.render(f'Lines:', True, (150, 150, 150)), (info_start_x, info_y_start + 60))
    surface.blit(score_font.render(f'{shot.line_count}', True, (255, 255, 255)), (info_start_x, info_y_start + 80))

    # Garbage Bar
    if shot.pending_garbage > 0:
        bar_max_height = config.rows * config.grid
        bar_y_start = draw_y
        pending_visual = min(shot.pending_garbage, 20) 
        bar_fill_ratio = pending_visual / 20.0
        bar_height = bar_max_height * bar_fill_ratio
        bar_x = off_x - config.GARBAGE_BAR_WIDTH - 5 + shake_x
        bar_y_fill = (bar_y_start + bar_max_height) - bar_height
        bar_color = (255, 50, 50)
        if getattr(shot, 'shake_timer', 0) > 0 and (shot.shake_timer // 2) % 2 == 0:
             bar_color = (255, 200, 200) 
        pg.draw.rect(surface, (40, 40, 40), (bar_x, bar_y_start, config.GARBAGE_BAR_WIDTH, bar_max_height))
        pg.draw.rect(surface, bar_color, (bar_x, bar_y_fill, config.GARBAGE_BAR_WIDTH, bar_height))
        pg.draw.rect(surface, (200, 200, 200), (bar_x, bar_y_start, config.GARBAGE_BAR_WIDTH, bar_max_height), 1)

    # --- 特效文字 ---
    # 1. Tetris Effect
    if getattr(shot, 'tetris_timer', 0) > 0:
        _draw_effect_text(surface, "TETRIS!", (255, 255, 255), shot.tetris_timer, draw_x, draw_y)

    #  All Clear Effect 
    if getattr(shot, 'all_clear_timer', 0) > 0:
        _draw_effect_text(surface, "ALL CLEAR!", (255, 215, 0), shot.all_clear_timer, draw_x, draw_y, offset_y=60)

    # Game Over Overlay
    if getattr(shot, 'game_over', False):
        s = pg.Surface((config.columns * config.grid, config.rows * config.grid))
        s.set_alpha(150)
        s.fill((0,0,0))
        surface.blit(s, (draw_x, draw_y))
        text = font.render("GAME OVER", True, (255, 50, 50))
        text_rect = text.get_rect(center=(draw_x + (config.columns * config.grid)//2, draw_y + (config.rows * config.grid)//2))
        surface.blit(text, text_rect)

    return surface