import pygame as pg
import sys
import os
import threading
import config
import settings
import network_utils
from ui import Button, Slider

# --- Settings Menu ---
# --- 設定選單 ---
 
def settings_menu(screen):
    pg.display.set_caption("Tetris Battle - Settings")
    font_title = pg.font.SysFont('Comic Sans MS', 50, bold=True)
    font_label = pg.font.SysFont('Arial', 24)
    
    btn_w, btn_h = 300, 60
    center_x = config.width // 2 - btn_w // 2
    center_y = config.height // 3
    
    btn_controls = Button(center_x, center_y + 80, btn_w, btn_h, "Controls", "CONTROLS", color=(50, 100, 200))

    btn_back = Button(center_x, center_y + 160, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    # Volume Slider
    slider_w, slider_h = 300, 10
    slider_x = config.width // 2 - slider_w // 2
    slider_y = center_y + 260
    volume_slider = Slider(slider_x, slider_y, slider_w, slider_h, 0.0, 1.0, settings.VOLUME)
    
    while True:
        status_text = "ON" if settings.SHOW_GHOST else "OFF"
        status_color = (50, 200, 50) if settings.SHOW_GHOST else (200, 50, 50)
        btn_ghost = Button(center_x, center_y, btn_w, btn_h, f"Ghost Piece: {status_text}", "TOGGLE_GHOST", color=status_color)
        
        buttons = [btn_ghost, btn_controls, btn_back]
        
        screen.fill(config.background_color)
        title_surf = font_title.render("SETTINGS", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//6))
        screen.blit(title_surf, title_rect)
        
        # Draw Volume Label
        vol_surf = font_label.render(f"Volume: {int(settings.VOLUME * 100)}%", True, (200, 200, 200))
        vol_rect = vol_surf.get_rect(center=(config.width//2, slider_y - 30))
        screen.blit(vol_surf, vol_rect)
        
        # Draw Slider
        volume_slider.draw(screen)
        
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            
            if volume_slider.handle_event(event):
                settings.VOLUME = volume_slider.value
                pg.mixer.music.set_volume(settings.VOLUME)
            
            if btn_ghost.is_clicked(event):
                settings.SHOW_GHOST = not settings.SHOW_GHOST
        
            # 新增 Controls 按鈕邏輯
            if btn_controls.is_clicked(event):
                controls_menu(screen)
                
            if btn_back.is_clicked(event):
                return 

        for btn in buttons: btn.draw(screen)
        pg.display.update()

# --- 控制設定選單 ---

def controls_menu(screen):
    pg.display.set_caption("Tetris Battle - Controls")
    font_title = pg.font.SysFont('Comic Sans MS', 40, bold=True)
    font_label = pg.font.SysFont('Arial', 24)
    
    btn_w, btn_h = 200, 40
    
    # 定義要設定的按鍵
    actions = [
        ("P1 Left", 'P1_LEFT'),
        ("P1 Right", 'P1_RIGHT'),
        ("P1 Down", 'P1_DOWN'),
        ("P1 Rotate CW", 'P1_ROTATE'),
        ("P1 Rotate CCW", 'P1_ROTATE_CCW'),
        ("P1 Drop", 'P1_DROP'),
        ("P2 Left", 'P2_LEFT'),
        ("P2 Right", 'P2_RIGHT'),
        ("P2 Down", 'P2_DOWN'),
        ("P2 Rotate CW", 'P2_ROTATE'),
        ("P2 Rotate CCW", 'P2_ROTATE_CCW'),
        ("P2 Drop", 'P2_DROP')
    ]
    
    waiting_for_key = None # 紀錄當前正在等待輸入的 action key (例如 'P1_LEFT')
    conflict_info = None # 用於儲存衝突資訊: {'key_code': int, 'new_action': str, 'old_action': str}
    
    while True:
        screen.fill(config.background_color)
        
        title_surf = font_title.render("CONTROLS CONFIG", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, 50))
        screen.blit(title_surf, title_rect)
        
        # 繪製按鍵設定列表
        start_y = 120
        col1_x = config.width // 4 + 50
        col2_x = config.width * 3 // 4 + 50
        
        buttons = []
        
        for i, (label, key_name) in enumerate(actions):
            is_p1 = i < 6
            x = col1_x if is_p1 else col2_x
            y = start_y + (i % 6) * 70
            
            # 顯示標籤
            label_surf = font_label.render(label, True, (200, 200, 200))
            label_rect = label_surf.get_rect(midright=(x - 110, y + btn_h//2))
            screen.blit(label_surf, label_rect)
            
            # 顯示按鍵按鈕
            current_key_code = settings.KEY_BINDINGS[key_name]
            key_text = pg.key.name(current_key_code).upper()
            
            btn_color = (100, 100, 100)
            if waiting_for_key == key_name:
                key_text = "PRESS KEY..."
                btn_color = (200, 50, 50)
            
            btn = Button(x - 100, y, btn_w, btn_h, key_text, key_name, color=btn_color)
            btn.draw(screen)
            buttons.append(btn)
            
        # Back Button
        btn_back = Button(config.width//2 - 100, config.height - 80, 200, 50, "Back", "BACK", color=(50, 50, 50))
        btn_back.draw(screen)
        buttons.append(btn_back)
        
        # --- 衝突提示視窗 ---
        if conflict_info:
            overlay = pg.Surface((config.width, config.height))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            # 提示框背景
            dialog_w, dialog_h = 500, 300
            dialog_x = config.width // 2 - dialog_w // 2
            dialog_y = config.height // 2 - dialog_h // 2
            pg.draw.rect(screen, (50, 50, 50), (dialog_x, dialog_y, dialog_w, dialog_h))
            pg.draw.rect(screen, (200, 200, 200), (dialog_x, dialog_y, dialog_w, dialog_h), 3)
            
            # 提示文字
            key_name_str = pg.key.name(conflict_info['key_code']).upper()
            old_action_label = next((l for l, k in actions if k == conflict_info['old_action']), conflict_info['old_action'])
            
            msg1 = font_label.render(f"Key '{key_name_str}' is already used by:", True, (255, 255, 255))
            msg2 = font_title.render(old_action_label, True, (255, 100, 100))
            msg3 = font_label.render("Do you want to swap keys?", True, (255, 255, 255))
            
            screen.blit(msg1, msg1.get_rect(center=(config.width//2, dialog_y + 60)))
            screen.blit(msg2, msg2.get_rect(center=(config.width//2, dialog_y + 120)))
            screen.blit(msg3, msg3.get_rect(center=(config.width//2, dialog_y + 180)))
            
            # Yes / No 按鈕
            btn_yes = Button(config.width//2 - 110, dialog_y + 220, 100, 50, "Yes", "YES", color=(50, 150, 50))
            btn_no = Button(config.width//2 + 10, dialog_y + 220, 100, 50, "No", "NO", color=(150, 50, 50))
            
            btn_yes.draw(screen)
            btn_no.draw(screen)
            
            conflict_buttons = [btn_yes, btn_no]
        
        pg.display.update()
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit(); sys.exit()
            
            # 處理衝突視窗事件
            if conflict_info:
                if event.type == pg.MOUSEBUTTONDOWN:
                    for btn in conflict_buttons:
                        if btn.is_clicked(event):
                            if btn.action_code == "YES":
                                # 執行交換
                                # 1. 取得目前 action 原本的 key (如果有的話)
                                current_action = conflict_info['new_action']
                                old_action = conflict_info['old_action']
                                target_key = conflict_info['key_code']
                                
                                # 2. 取得 current_action 原本的 key
                                original_key_of_current = settings.KEY_BINDINGS[current_action]
                                
                                # 3. 交換
                                settings.KEY_BINDINGS[current_action] = target_key
                                settings.KEY_BINDINGS[old_action] = original_key_of_current
                                
                                conflict_info = None
                                waiting_for_key = None
                                
                            elif btn.action_code == "NO":
                                # 取消
                                conflict_info = None
                                waiting_for_key = None # 也可以保持 waiting 狀態讓使用者重選，這裡選擇取消
                continue # 衝突視窗開啟時，不處理其他事件

            if event.type == pg.MOUSEBUTTONDOWN:
                if waiting_for_key:
                    # 如果正在等待按鍵，點擊滑鼠取消等待
                    waiting_for_key = None
                    continue
                    
                for btn in buttons:
                    if btn.is_clicked(event):
                        if btn.action_code == "BACK":
                            return
                        else:
                            # 點擊了某個按鍵設定按鈕
                            waiting_for_key = btn.action_code
                            
            if event.type == pg.KEYDOWN:
                if waiting_for_key:
                    if event.key != pg.K_ESCAPE:
                        # 檢查衝突
                        conflict_action = None
                        for act, code in settings.KEY_BINDINGS.items():
                            if code == event.key and act != waiting_for_key:
                                conflict_action = act
                                break
                        
                        if conflict_action:
                            # 發現衝突，顯示提示
                            conflict_info = {
                                'key_code': event.key,
                                'new_action': waiting_for_key,
                                'old_action': conflict_action
                            }
                        else:
                            # 無衝突，直接綁定
                            settings.KEY_BINDINGS[waiting_for_key] = event.key
                            waiting_for_key = None
                    else:
                        waiting_for_key = None

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
    text_rect = text_surf.get_rect(center=(config.width//2, config.height//6))
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
    
    # Load background
    bg_image = None
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        bg_path = os.path.join(base_path, 'background.png')
        if os.path.exists(bg_path):
            bg_image = pg.image.load(bg_path).convert()
            bg_image = pg.transform.scale(bg_image, (config.width, config.height))
    except Exception as e:
        print(f"Failed to load background: {e}")

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
        if bg_image:
            screen.blit(bg_image, (0, 0))
        else:
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

# --- LAN Menu ---
def lan_menu(screen, font):
    pg.display.set_caption("LAN Battle Setup")
    btn_w, btn_h = 300, 60
    center_x = config.width // 2 - btn_w // 2
    start_y = config.height // 4
    btn_host = Button(center_x, start_y, btn_w, btn_h, "Host Game", "HOST", color=(50, 150, 50))
    btn_join = Button(center_x, start_y + 80, btn_w, btn_h, "Join Game", "JOIN", color=(50, 100, 200))
    btn_back = Button(center_x, start_y + 300, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    # Player Count Buttons (for Host)
    # Center them properly
    cx = config.width // 2
    btn_2p = Button(cx - 110, start_y + 330, 60, 60, "2P", "2P", color=(100, 100, 100))
    btn_3p = Button(cx - 30, start_y + 330, 60, 60, "3P", "3P", color=(100, 100, 100))
    btn_4p = Button(cx + 50, start_y + 330, 60, 60, "4P", "4P", color=(100, 100, 100))
    
    player_btns = [btn_2p, btn_3p, btn_4p]
    selected_players = 2
    btn_2p.color = (50, 200, 50) # Default selected
    
    # Adjust Back button position
    btn_back = Button(center_x, start_y + 430, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    buttons = [btn_host, btn_join, btn_back] + player_btns
    ip_text = "127.0.0.1"
    input_active = False
    # Make input box bigger
    input_w, input_h = 400, 60
    input_rect = pg.Rect(config.width // 2 - input_w // 2, start_y + 160, input_w, input_h)
    
    # Port Input
    port_text = "5555"
    port_active = False
    port_w, port_h = 100, 60
    port_rect = pg.Rect(input_rect.right + 20, start_y + 160, port_w, port_h)
    
    net_mgr = network_utils.NetworkManager()
    all_ips = net_mgr.get_all_ips()
    current_ip_idx = 0
    my_local_ip = all_ips[current_ip_idx]
    
    btn_cycle_ip = Button(config.width//2 + 250, config.height//8 + 50, 100, 40, "Next IP", "CYCLE_IP", color=(100, 100, 100))
    
    while True:
        screen.fill(config.background_color)
        title_surf = pg.font.SysFont('Comic Sans MS', 40, bold=True).render("LAN SETUP", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//8))
        screen.blit(title_surf, title_rect)

        # Display Local IP
        ip_display_surf = font.render(f"Your IP: {my_local_ip}", True, (100, 255, 100))
        ip_display_rect = ip_display_surf.get_rect(center=(config.width//2, config.height//8 + 50))
        screen.blit(ip_display_surf, ip_display_rect)
        
        if len(all_ips) > 1:
            btn_cycle_ip.draw(screen)
        
        # Draw Input Box (IP)
        color = (255, 255, 255) if input_active else (150, 150, 150)
        pg.draw.rect(screen, color, input_rect, 2)
        text_surface = font.render(ip_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(midleft=(input_rect.x + 10, input_rect.centery))
        screen.blit(text_surface, text_rect)
        label_surf = font.render("Host IP:", True, (200, 200, 200))
        label_rect = label_surf.get_rect(midright=(input_rect.x - 10, input_rect.centery))
        screen.blit(label_surf, label_rect)
        
        # Draw Input Box (Port)
        color_p = (255, 255, 255) if port_active else (150, 150, 150)
        pg.draw.rect(screen, color_p, port_rect, 2)
        port_surf = font.render(port_text, True, (255, 255, 255))
        port_txt_rect = port_surf.get_rect(midleft=(port_rect.x + 10, port_rect.centery))
        screen.blit(port_surf, port_txt_rect)
        port_lbl = font.render("Port:", True, (200, 200, 200))
        port_lbl_rect = port_lbl.get_rect(midright=(port_rect.x - 10, port_rect.centery))
        screen.blit(port_lbl, port_lbl_rect)

        cnt_surf = font.render("Max Players:", True, (200, 200, 200))
        screen.blit(cnt_surf, cnt_surf.get_rect(center=(center_x + btn_w//2, start_y + 280)))
        
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            if event.type == pg.MOUSEBUTTONDOWN:
                if input_rect.collidepoint(event.pos): input_active = True; port_active = False
                elif port_rect.collidepoint(event.pos): port_active = True; input_active = False
                else: input_active = False; port_active = False
                
                # Check cycle IP button
                if len(all_ips) > 1 and btn_cycle_ip.is_clicked(event):
                    current_ip_idx = (current_ip_idx + 1) % len(all_ips)
                    my_local_ip = all_ips[current_ip_idx]
                
                for btn in buttons:
                    if btn.is_clicked(event):
                        if btn.action_code == "BACK": return None, None
                        elif btn.action_code in ["2P", "3P", "4P"]:
                            selected_players = int(btn.action_code[0])
                            for b in player_btns: b.color = (100, 100, 100)
                            btn.color = (50, 200, 50)
                        elif btn.action_code == "HOST":
                            # Use the selected IP for display in lobby
                            local_ip = my_local_ip 
                            try:
                                target_port = int(port_text)
                            except:
                                target_port = 5555
                                
                            # Bind to the specific IP selected by user
                            host_thread = threading.Thread(target=net_mgr.host_game, args=(target_port, selected_players, local_ip), daemon=True)
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
                                txt_port = font.render(f"Port: {target_port}", True, (200, 200, 200))
                                screen.blit(txt_port, txt_port.get_rect(center=(config.width//2, info_y + 30)))
                                txt_count = font.render(f"Players Connected: {current_players} / {selected_players}", True, (255, 255, 255))
                                screen.blit(txt_count, txt_count.get_rect(center=(config.width//2, info_y + 60)))
                                txt_hint = pg.font.SysFont('Arial', 20).render("Share IP & Port. Check Firewall if join fails.", True, (150, 150, 150))
                                screen.blit(txt_hint, txt_hint.get_rect(center=(config.width//2, info_y + 100)))
                                btn_start.draw(screen)
                                pg.display.update(); clock.tick(30)
                        elif btn.action_code == "JOIN":
                            try:
                                target_port = int(port_text)
                            except:
                                target_port = 5555
                            join_thread = threading.Thread(target=net_mgr.join_game, args=(ip_text, target_port), daemon=True)
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
                                        hint_surf = pg.font.SysFont('Arial', 20).render("Check Host IP or Windows Firewall", True, (200, 200, 200))
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
                elif port_active:
                    if event.key == pg.K_RETURN: port_active = False
                    elif event.key == pg.K_BACKSPACE: port_text = port_text[:-1]
                    elif event.unicode.isdigit(): port_text += event.unicode
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
    start_y = config.height // 3 - 40
    
    btn_weight = Button(center_x, start_y, btn_w, btn_h, "Weighted AI (8-Param)", "WEIGHT", color=(50, 100, 200))
    btn_expert = Button(center_x, start_y + 80, btn_w, btn_h, "Expert AI (Fast)", "EXPERT", color=(200, 50, 50))
    btn_back = Button(center_x, start_y + 240, btn_w, btn_h, "Back", "BACK", color=(100, 100, 100))
    
    font_title = pg.font.SysFont('Comic Sans MS', 40, bold=True)
    
    while True:
        screen.fill(config.background_color)
        title_surf = font_title.render("CHOOSE OPPONENT", True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(config.width//2, config.height//6))
        screen.blit(title_surf, title_rect)
        
        # AI Speed Toggle Button
        speed_text = ["Slow", "Normal", "Fast", "Instant"][settings.AI_SPEED_LEVEL - 1]
        btn_speed = Button(center_x, start_y + 160, btn_w, btn_h, f"AI Speed: {speed_text}", "TOGGLE_SPEED", color=(150, 150, 50))
        
        buttons = [btn_weight, btn_expert, btn_speed, btn_back]
        
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            
            if btn_speed.is_clicked(event):
                settings.AI_SPEED_LEVEL = (settings.AI_SPEED_LEVEL % 4) + 1
            
            for btn in buttons:
                if btn.is_clicked(event):
                    if btn.action_code != "TOGGLE_SPEED":
                        return btn.action_code 
        
        for btn in buttons: btn.draw(screen)
        pg.display.update()
                    
        for btn in buttons: btn.draw(screen)
        pg.display.update()