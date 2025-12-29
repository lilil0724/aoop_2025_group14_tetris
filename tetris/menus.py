import pygame as pg
import sys
import threading
import config
import settings
import network_utils
from ui import Button

# --- Settings Menu ---
def settings_menu(screen):
    pg.display.set_caption("Tetris Battle - Settings")
    font_title = pg.font.SysFont('Comic Sans MS', 50, bold=True)
    btn_w, btn_h = 300, 60
    center_x = config.width // 2 - btn_w // 2
    center_y = config.height // 3
    btn_back = Button(center_x, center_y + 160, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    while True:
        status_text = "ON" if settings.SHOW_GHOST else "OFF"
        status_color = (50, 200, 50) if settings.SHOW_GHOST else (200, 50, 50)
        btn_ghost = Button(center_x, center_y, btn_w, btn_h, f"Ghost Piece: {status_text}", "TOGGLE_GHOST", color=status_color)
        
        # 定義速度顯示文字
        speed_labels = {1: "Slow", 2: "Normal", 3: "Fast", 4: "God (Instant)"}
        current_label = speed_labels.get(settings.AI_SPEED_LEVEL, "Normal")
        speed_colors = {1: (50, 200, 50), 2: (200, 200, 50), 3: (200, 100, 50), 4: (200, 50, 50)}
        current_color = speed_colors.get(settings.AI_SPEED_LEVEL, (100, 100, 100))
        btn_speed = Button(center_x, center_y + 80, btn_w, btn_h, f"AI Speed: {current_label}", "TOGGLE_SPEED", color=current_color)
        
        btn_keybind = Button(center_x, center_y + 160, btn_w, btn_h, "Key Bindings", "KEYBIND", color=(50, 100, 150))
        btn_back = Button(center_x, center_y + 240, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
        
        buttons = [btn_ghost, btn_speed, btn_keybind, btn_back]
        
        screen.fill(config.background_color)
        title_surf = font_title.render("SETTINGS", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//6))
        screen.blit(title_surf, title_rect)
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            if btn_ghost.is_clicked(event): settings.SHOW_GHOST = not settings.SHOW_GHOST
            if btn_speed.is_clicked(event): settings.AI_SPEED_LEVEL = (settings.AI_SPEED_LEVEL % 4) + 1
            if btn_keybind.is_clicked(event): keybind_menu(screen)
            if btn_back.is_clicked(event): return 
        for btn in buttons: btn.draw(screen)
        pg.display.update()

def keybind_menu(screen):
    pg.display.set_caption("Key Bindings")
    font_title = pg.font.SysFont('Comic Sans MS', 40, bold=True)
    font_label = pg.font.SysFont('Arial', 20)
    
    # Helper to get key name
    def get_key_name(k):
        return pg.key.name(k).upper()

    # Helper to check conflict
    def is_key_used(new_key):
        for p in settings.KEYBINDS:
            for action, k in settings.KEYBINDS[p].items():
                if k == new_key:
                    return True
        return False

    waiting_for_key = None # (player, action)
    warning_msg = ""
    warning_timer = 0
    
    btn_back = Button(config.width//2 - 100, config.height - 80, 200, 50, "Back", "BACK", color=(100, 100, 100))
    
    while True:
        screen.fill(config.background_color)
        title_surf = font_title.render("KEY BINDINGS", True, (255, 255, 255))
        screen.blit(title_surf, title_surf.get_rect(center=(config.width//2, 50)))
        
        # Draw P1 Column
        p1_x = config.width // 4
        p1_title = font_title.render("PLAYER 1", True, (100, 200, 255))
        screen.blit(p1_title, p1_title.get_rect(center=(p1_x, 120)))
        
        # Draw P2 Column
        p2_x = 3 * config.width // 4
        p2_title = font_title.render("PLAYER 2", True, (255, 100, 100))
        screen.blit(p2_title, p2_title.get_rect(center=(p2_x, 120)))
        
        # Draw Buttons
        buttons = [btn_back]
        
        y_start = 180
        gap = 60
        
        # Actions to bind
        common_actions = ['ROTATE', 'SOFT_DROP', 'LEFT', 'RIGHT', 'HARD_DROP']
        
        for i, action in enumerate(common_actions):
            y = y_start + i * gap
            
            # P1 Button
            key_p1 = settings.KEYBINDS['P1'].get(action)
            txt_p1 = f"{action}: {get_key_name(key_p1)}" if key_p1 else f"{action}: N/A"
            color_p1 = (50, 150, 200)
            if waiting_for_key == ('P1', action):
                txt_p1 = "PRESS KEY..."
                color_p1 = (255, 200, 50)
            
            btn_p1 = Button(p1_x - 120, y, 240, 40, txt_p1, f"P1_{action}", color=color_p1)
            btn_p1.font = font_label
            buttons.append(btn_p1)
            
            # P2 Button
            key_p2 = settings.KEYBINDS['P2'].get(action)
            txt_p2 = f"{action}: {get_key_name(key_p2)}" if key_p2 else f"{action}: N/A"
            color_p2 = (200, 100, 100)
            if waiting_for_key == ('P2', action):
                txt_p2 = "PRESS KEY..."
                color_p2 = (255, 200, 50)
                
            btn_p2 = Button(p2_x - 120, y, 240, 40, txt_p2, f"P2_{action}", color=color_p2)
            btn_p2.font = font_label
            buttons.append(btn_p2)

        # Extra P1 keys (CCW, PVP Hard Drop)
        # Just hardcode positions for now or add to list
        
        # Handle events
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            
            if waiting_for_key:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        waiting_for_key = None
                        warning_msg = ""
                    else:
                        # Check conflict
                        if is_key_used(event.key):
                            warning_msg = f"Key '{get_key_name(event.key)}' is already in use!"
                            warning_timer = 60
                        else:
                            player, action = waiting_for_key
                            settings.KEYBINDS[player][action] = event.key
                            waiting_for_key = None
                            warning_msg = ""
            else:
                if btn_back.is_clicked(event): return
                for btn in buttons:
                    if btn.is_clicked(event):
                        if btn.action_code.startswith("P1_") or btn.action_code.startswith("P2_"):
                            parts = btn.action_code.split("_", 1)
                            waiting_for_key = (parts[0], parts[1])
                            warning_msg = ""
                            
        for btn in buttons: btn.draw(screen)
        
        if warning_msg and warning_timer > 0:
            warn_surf = font_label.render(warning_msg, True, (255, 50, 50))
            screen.blit(warn_surf, warn_surf.get_rect(center=(config.width//2, config.height - 140)))
            warning_timer -= 1
        
        pg.display.update()

# --- Pause Menu ---
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

# --- Main Menu ---
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
        
        _draw_controls_info(screen, config.width - 320, config.height - 250)

        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            for btn in buttons:
                if btn.is_clicked(event): return btn.action_code
        for btn in buttons: btn.draw(screen)
        pg.display.update()

def _draw_controls_info(screen, x, y):
    font_title = pg.font.SysFont('Arial', 24, bold=True)
    font_text = pg.font.SysFont('Arial', 20)
    bg_rect = pg.Rect(x - 20, y - 20, 300, 220)
    s = pg.Surface((bg_rect.w, bg_rect.h), pg.SRCALPHA)
    s.fill((0, 0, 0, 100))
    screen.blit(s, (bg_rect.x, bg_rect.y))
    pg.draw.rect(screen, (255, 255, 255), bg_rect, 2)
    
    title = font_title.render("CONTROLS (Solo/PvE)", True, (255, 215, 0))
    screen.blit(title, (x, y))
    
    lines = [
        ("Move", "A / D"),
        ("Soft Drop", "S"),
        ("Hard Drop", "SPACE"),
        ("Rotate CW", "W"),
        ("Rotate CCW", "L"),
        ("PVP P2", "Arrows / R-SHIFT")
    ]
    
    line_height = 28
    current_y = y + 35
    for label, key in lines:
        lbl_surf = font_text.render(label, True, (200, 200, 200))
        key_surf = font_text.render(key, True, (100, 255, 100))
        screen.blit(lbl_surf, (x, current_y))
        key_rect = key_surf.get_rect(topright=(x + 260, current_y))
        screen.blit(key_surf, key_rect)
        current_y += line_height

# --- LAN Menu ---
def lan_menu(screen, font):
    pg.display.set_caption("LAN Battle Setup")
    btn_w, btn_h = 300, 60
    center_x = config.width // 2 - btn_w // 2
    start_y = config.height // 4
    btn_host = Button(center_x, start_y, btn_w, btn_h, "Host Game", "HOST", color=(50, 150, 50))
    btn_join = Button(center_x, start_y + 80, btn_w, btn_h, "Join Game", "JOIN", color=(50, 100, 200))
    btn_back = Button(center_x, start_y + 400, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    btn_2p = Button(center_x - 110, start_y + 300, 60, 60, "2P", "2P", color=(100, 100, 100))
    btn_3p = Button(center_x, start_y + 300, 60, 60, "3P", "3P", color=(100, 100, 100))
    btn_4p = Button(center_x + 110, start_y + 300, 60, 60, "4P", "4P", color=(100, 100, 100))
    player_btns = [btn_2p, btn_3p, btn_4p]
    selected_players = 2
    btn_2p.color = (50, 200, 50)
    
    buttons = [btn_host, btn_join, btn_back] + player_btns
    ip_text = "127.0.0.1"
    input_active = False
    input_rect = pg.Rect(center_x, start_y + 160, btn_w, 40)
    net_mgr = network_utils.NetworkManager()
    
    while True:
        screen.fill(config.background_color)
        title_surf = pg.font.SysFont('Comic Sans MS', 40, bold=True).render("LAN SETUP", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//8))
        screen.blit(title_surf, title_rect)
        color = (255, 255, 255) if input_active else (150, 150, 150)
        pg.draw.rect(screen, color, input_rect, 2)
        text_surface = font.render(ip_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(midleft=(input_rect.x + 10, input_rect.centery))
        screen.blit(text_surface, text_rect)
        label_surf = font.render("Host IP:", True, (200, 200, 200))
        label_rect = label_surf.get_rect(midright=(input_rect.x - 10, input_rect.centery))
        screen.blit(label_surf, label_rect)
        cnt_surf = font.render("Max Players:", True, (200, 200, 200))
        screen.blit(cnt_surf, cnt_surf.get_rect(center=(center_x + btn_w//2, start_y + 280)))
        
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            if event.type == pg.MOUSEBUTTONDOWN:
                if input_rect.collidepoint(event.pos): input_active = True
                else: input_active = False
                for btn in buttons:
                    if btn.is_clicked(event):
                        if btn.action_code == "BACK": return None, None
                        elif btn.action_code in ["2P", "3P", "4P"]:
                            selected_players = int(btn.action_code[0])
                            for b in player_btns: b.color = (100, 100, 100)
                            btn.color = (50, 200, 50)
                        elif btn.action_code == "HOST":
                            local_ip = net_mgr.get_local_ip()
                            host_thread = threading.Thread(target=net_mgr.host_game, args=(5555, selected_players), daemon=True)
                            host_thread.start()
                            waiting = True
                            clock = pg.time.Clock()
                            btn_start = Button(center_x, start_y + 400, btn_w, btn_h, "Start Game", "START", color=(50, 200, 50))
                            while waiting:
                                current_players = len(net_mgr.clients) + 1
                                for e in pg.event.get():
                                    if e.type == pg.QUIT: net_mgr.close(); pg.quit(); sys.exit()
                                    if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE: net_mgr.close(); waiting = False; net_mgr = network_utils.NetworkManager()
                                    if btn_start.is_clicked(e): net_mgr.start_game(); return "LAN", net_mgr
                                screen.fill(config.background_color)
                                title_s = font.render("Lobby - Waiting for Players", True, (255, 255, 255))
                                screen.blit(title_s, title_s.get_rect(center=(config.width//2, config.height//6)))
                                info_y = config.height//3
                                txt_ip = font.render(f"Host IP: {local_ip}", True, (255, 215, 0))
                                screen.blit(txt_ip, txt_ip.get_rect(center=(config.width//2, info_y)))
                                txt_count = font.render(f"Players Connected: {current_players} / {selected_players}", True, (255, 255, 255))
                                screen.blit(txt_count, txt_count.get_rect(center=(config.width//2, info_y + 60)))
                                txt_hint = pg.font.SysFont('Arial', 20).render("Share the IP with your friends to join.", True, (150, 150, 150))
                                screen.blit(txt_hint, txt_hint.get_rect(center=(config.width//2, info_y + 100)))
                                btn_start.draw(screen)
                                pg.display.update(); clock.tick(30)
                        elif btn.action_code == "JOIN":
                            join_thread = threading.Thread(target=net_mgr.join_game, args=(ip_text,), daemon=True)
                            join_thread.start()
                            waiting = True
                            clock = pg.time.Clock()
                            while waiting:
                                if net_mgr.connected:
                                    if net_mgr.game_started: return "LAN", net_mgr
                                    screen.fill(config.background_color)
                                    txt1 = font.render("Connected! Waiting for Host to Start...", True, (50, 255, 50))
                                    txt2 = font.render("Please wait...", True, (200, 200, 200))
                                    screen.blit(txt1, txt1.get_rect(center=(config.width//2, config.height//2 - 20)))
                                    screen.blit(txt2, txt2.get_rect(center=(config.width//2, config.height//2 + 30)))
                                    for e in pg.event.get():
                                        if e.type == pg.QUIT: net_mgr.close(); pg.quit(); sys.exit()
                                        if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE: net_mgr.close(); waiting = False; net_mgr = network_utils.NetworkManager()
                                    pg.display.update(); clock.tick(30); continue
                                if not join_thread.is_alive():
                                    waiting = False
                                    err_start = pg.time.get_ticks()
                                    while pg.time.get_ticks() - err_start < 3000:
                                        screen.fill(config.background_color)
                                        err_surf = font.render("Connection Failed!", True, (255, 50, 50))
                                        hint_surf = pg.font.SysFont('Arial', 20).render("Check IP or Firewall settings", True, (200, 200, 200))
                                        screen.blit(err_surf, err_surf.get_rect(center=(config.width//2, config.height//2 - 20)))
                                        screen.blit(hint_surf, hint_surf.get_rect(center=(config.width//2, config.height//2 + 30)))
                                        pg.display.update(); pg.event.pump()
                                    net_mgr = network_utils.NetworkManager()
                                for e in pg.event.get():
                                    if e.type == pg.QUIT: net_mgr.close(); pg.quit(); sys.exit()
                                    if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE: net_mgr.close(); waiting = False; net_mgr = network_utils.NetworkManager()
                                screen.fill(config.background_color)
                                txt1 = font.render(f"Connecting to {ip_text}...", True, (255, 255, 255))
                                txt2 = font.render("Press ESC to Cancel", True, (150, 150, 150))
                                screen.blit(txt1, txt1.get_rect(center=(config.width//2, config.height//2)))
                                screen.blit(txt2, txt2.get_rect(center=(config.width//2, config.height//2 + 60)))
                                pg.display.update(); clock.tick(30)
            if event.type == pg.KEYDOWN:
                if input_active:
                    if event.key == pg.K_RETURN: input_active = False
                    elif event.key == pg.K_BACKSPACE: ip_text = ip_text[:-1]
                    else: ip_text += event.unicode
        for btn in buttons: btn.draw(screen)
        pg.display.update()

# --- Game Over Screen (修正為接收 List) ---
def game_over_screen(screen, results):
    pg.display.set_caption("Game Over")
    
    font_title = pg.font.SysFont('Comic Sans MS', 50, bold=True)
    font_label = pg.font.SysFont('Arial', 24)
    font_val = pg.font.SysFont('Arial', 30, bold=True)
    
    btn_restart = Button(config.width//2 - 100, config.height - 150, 200, 60, "Play Again", "RESTART")
    btn_menu = Button(config.width//2 - 100, config.height - 70, 200, 60, "Main Menu", "MENU", color=(150, 50, 50))
    buttons = [btn_restart, btn_menu]
    
    while True:
        s = pg.Surface((config.width, config.height))
        s.set_alpha(10) 
        screen.fill((20, 20, 20))
        
        title_surf = font_title.render("GAME FINISHED", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, 80))
        screen.blit(title_surf, title_rect)
        
        count = len(results)
        card_w = 300
        card_h = 250
        gap = 50
        total_w = count * card_w + (count - 1) * gap
        start_x = (config.width - total_w) // 2
        start_y = config.height // 2 - 100
        
        for i, data in enumerate(results):
            x = start_x + i * (card_w + gap)
            y = start_y
            
            is_winner = data['is_winner']
            if is_winner:
                border_color = (255, 215, 0) # Gold
                header_text = "WINNER"
                header_color = (255, 215, 0)
            else:
                border_color = (100, 100, 100) # Grey
                header_text = "LOSER"
                header_color = (200, 50, 50) # Red
            
            card_rect = pg.Rect(x, y, card_w, card_h)
            pg.draw.rect(screen, (40, 40, 40), card_rect, border_radius=15)
            pg.draw.rect(screen, border_color, card_rect, 3, border_radius=15)
            
            head_surf = font_title.render(header_text, True, header_color)
            head_surf = pg.transform.smoothscale(head_surf, (int(head_surf.get_width()*0.8), int(head_surf.get_height()*0.8)))
            head_rect = head_surf.get_rect(center=(x + card_w//2, y + 40))
            screen.blit(head_surf, head_rect)
            
            name_surf = font_label.render(data['name'], True, (200, 200, 200))
            name_rect = name_surf.get_rect(center=(x + card_w//2, y + 90))
            screen.blit(name_surf, name_rect)
            
            pg.draw.line(screen, (80, 80, 80), (x + 20, y + 110), (x + card_w - 20, y + 110), 1)
            
            score_lbl = font_label.render("Score", True, (150, 150, 150))
            score_val = font_val.render(str(data['score']), True, (255, 255, 255))
            screen.blit(score_lbl, (x + 30, y + 130))
            screen.blit(score_val, score_val.get_rect(topright=(x + card_w - 30, y + 130)))
            
            lines_lbl = font_label.render("Lines", True, (150, 150, 150))
            lines_val = font_val.render(str(data['lines']), True, (255, 255, 255))
            screen.blit(lines_lbl, (x + 30, y + 180))
            screen.blit(lines_val, lines_val.get_rect(topright=(x + card_w - 30, y + 180)))

        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            for btn in buttons:
                if btn.is_clicked(event): return btn.action_code
        for btn in buttons: btn.draw(screen)
        pg.display.update()

def ai_selection_menu(screen, font):
    pg.display.set_caption("Select AI Opponent")
    
    btn_w, btn_h = 320, 60
    center_x = config.width // 2 - btn_w // 2
    start_y = config.height // 3
    
    btn_weight = Button(center_x, start_y, btn_w, btn_h, "Weighted AI (8-Param)", "WEIGHT", color=(50, 100, 200))
    btn_expert = Button(center_x, start_y + 80, btn_w, btn_h, "Expert AI (Fast)", "EXPERT", color=(200, 50, 50))
    btn_back = Button(center_x, start_y + 200, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    buttons = [btn_weight, btn_expert, btn_back]
    font_title = pg.font.SysFont('Comic Sans MS', 40, bold=True)
    
    while True:
        screen.fill(config.background_color)
        title_surf = font_title.render("CHOOSE OPPONENT", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//6))
        screen.blit(title_surf, title_rect)
        
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            for btn in buttons:
                if btn.is_clicked(event): return btn.action_code 
                    
        for btn in buttons: btn.draw(screen)
        pg.display.update()