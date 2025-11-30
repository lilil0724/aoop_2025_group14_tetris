import pygame as pg
import sys
import copy
import random
import time
import config
import pieces
import shots
import Handler
import settings
from ui import draw_player_ui_surface
from ai_heuristic import get_ai_move_heuristic
from ai_weighted import WeightedAI
from menus import pause_menu

def run_game(screen, clock, font, mode, ai_mode=None, net_mgr=None):
    
    class PlayerContext:
        def __init__(self, is_local=False, is_ai=False, name="Player"):
            self.shot = shots.Shot()
            self.shot.tetris_timer = 0
            self.piece = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
            self.next_piece = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
            self.game_over = False
            self.is_local = is_local
            self.is_ai = is_ai
            self.name = name
            
            self.counter = 0
            self.key_ticker = {k: 0 for k in [pg.K_a, pg.K_s, pg.K_d, pg.K_w, pg.K_LEFT, pg.K_RIGHT, pg.K_DOWN, pg.K_UP]}
            
            # AI 狀態
            self.ai_target_move = None 
            self.ai_act_timer = 0
            self.ai_act_interval = 5 
            self.ai_agent = None # 用來儲存 AI 物件或標記

    players = {}
    my_id = 0
    
    # --- Initialization ---
    if mode == 'LAN':
        if net_mgr.is_server:
            my_id = 0
            players[0] = PlayerContext(is_local=True, name="Host (You)")
        else:
            start_wait = time.time()
            while net_mgr.my_id is None:
                if time.time() - start_wait > 5: return "MENU"
                time.sleep(0.1)
            my_id = net_mgr.my_id
            players[my_id] = PlayerContext(is_local=True, name=f"Player {my_id} (You)")

    elif mode == 'SOLO':
        players[0] = PlayerContext(is_local=True, name="Player 1")
        
    elif mode == 'PVP':
        players[0] = PlayerContext(is_local=True, name="Player 1")
        players[1] = PlayerContext(is_local=True, name="Player 2")
        
    elif mode == 'PVE':
        players[0] = PlayerContext(is_local=True, name="Player 1")
        
        # 讀取設定值
        speed_map = {
            1: 15,  # Slow
            2: 8,   # Normal
            3: 3,   # Fast
            4: 0    # God
        }
        selected_speed_delay = speed_map.get(settings.AI_SPEED_LEVEL, 8)
        
        if ai_mode == 'WEIGHT':
            players[1] = PlayerContext(is_local=False, is_ai=True, name="Weighted AI")
            players[1].ai_agent = WeightedAI() 
            players[1].ai_act_interval = selected_speed_delay
            print(f"Loaded Weighted AI. Speed Level: {settings.AI_SPEED_LEVEL}")
            
        elif ai_mode == 'EXPERT': 
            # [修復 2] 這裡原本錯誤呼叫了 get_ai_move_heuristic() 導致閃退
            # 我們只需要標記它是 EXPERT，實際運算在主迴圈直接呼叫函式
            players[1] = PlayerContext(is_local=False, is_ai=True, name="Expert AI")
            players[1].ai_agent = "EXPERT_FUNC" 
            players[1].ai_act_interval = selected_speed_delay
            print(f"Loaded Expert AI. Speed Level: {settings.AI_SPEED_LEVEL}")
            
        else:
            players[1] = PlayerContext(is_local=False, is_ai=True, name="Basic AI")
            players[1].ai_act_interval = selected_speed_delay

    running = True
    paused = False
    send_timer = 0 
    
    while running:
        if paused:
            action = pause_menu(screen)
            if action == "RESUME": paused = False
            elif action == "RESTART": return "RESTART"
            elif action == "MENU": return "MENU"
            clock.tick(30); continue

        # --- Network Sync (LAN) ---
        if mode == 'LAN':
            if not net_mgr.connected: return "MENU"
            remote_data = net_mgr.get_latest_data()
            for pid, data in remote_data.items():
                if pid == my_id: continue
                if pid not in players: players[pid] = PlayerContext(name=f"Player {pid}")
                p = players[pid]
                p.shot.status = data['status']
                p.shot.color = data['color']
                p.shot.score = data['score']
                p.shot.line_count = data['lines']
                p.shot.pending_garbage = data.get('pending_garbage', 0)
                p.piece.x = data['piece_x']
                p.piece.y = data['piece_y']
                p.piece.shape = data['piece_shape']
                p.piece.rotation = data['piece_rot']
                p.piece.color = data['piece_color']
                p.next_piece.shape = data['next_piece_shape']
                p.next_piece.color = data['next_piece_color']
                p.game_over = data['game_over']
            diff = net_mgr.get_garbage_diff()
            if diff > 0: players[my_id].shot.pending_garbage += diff
            send_timer = (send_timer + 1) % 3
            if send_timer == 0:
                me = players[my_id]
                local_data = {
                    'status': me.shot.status, 'color': me.shot.color, 'score': me.shot.score,
                    'lines': me.shot.line_count, 'pending_garbage': me.shot.pending_garbage,
                    'piece_x': me.piece.x, 'piece_y': me.piece.y, 'piece_shape': me.piece.shape,
                    'piece_rot': me.piece.rotation, 'piece_color': me.piece.color,
                    'next_piece_shape': me.next_piece.shape, 'next_piece_color': me.next_piece.color,
                    'game_over': me.game_over
                }
                net_mgr.send(local_data)

        # --- Event Handling ---
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE: paused = True
                p1 = players[my_id]
                if not p1.game_over:
                    if event.key == pg.K_w: Handler.rotate(p1.shot, p1.piece)
                    if event.key == pg.K_s: p1.key_ticker[pg.K_s] = 13; Handler.drop(p1.shot, p1.piece)
                    if event.key == pg.K_a: p1.key_ticker[pg.K_a] = 13; Handler.moveLeft(p1.shot, p1.piece)
                    if event.key == pg.K_d: p1.key_ticker[pg.K_d] = 13; Handler.moveRight(p1.shot, p1.piece)
                    
                    # [修改 4] 按鍵綁定: 除了 PVP 外，Space 也可以做 Instant Drop
                    is_instant_drop = False
                    if event.key == pg.K_LSHIFT: is_instant_drop = True
                    if mode != 'PVP' and event.key == pg.K_SPACE: is_instant_drop = True
                    
                    if is_instant_drop:
                        Handler.instantDrop(p1.shot, p1.piece)
                
                if mode == 'PVP':
                    p2 = players[1]
                    if not p2.game_over:
                        if event.key == pg.K_UP: Handler.rotate(p2.shot, p2.piece)
                        if event.key == pg.K_DOWN: p2.key_ticker[pg.K_DOWN] = 13; Handler.drop(p2.shot, p2.piece)
                        if event.key == pg.K_LEFT: p2.key_ticker[pg.K_LEFT] = 13; Handler.moveLeft(p2.shot, p2.piece)
                        if event.key == pg.K_RIGHT: p2.key_ticker[pg.K_RIGHT] = 13; Handler.moveRight(p2.shot, p2.piece)
                        if event.key == pg.K_RSHIFT: Handler.instantDrop(p2.shot, p2.piece)

        # --- DAS ---
        keys = pg.key.get_pressed()
        def do_das(p, k_l, k_r, k_d):
            if p.game_over: return
            if keys[k_l] and p.key_ticker[k_l] == 0: p.key_ticker[k_l] = 6; Handler.moveLeft(p.shot, p.piece)
            if keys[k_r] and p.key_ticker[k_r] == 0: p.key_ticker[k_r] = 6; Handler.moveRight(p.shot, p.piece)
            if keys[k_d] and p.key_ticker[k_d] == 0: p.key_ticker[k_d] = 6; Handler.drop(p.shot, p.piece)
            for k in p.key_ticker:
                if p.key_ticker[k] > 0: p.key_ticker[k] -= 1
        do_das(players[my_id], pg.K_a, pg.K_d, pg.K_s)
        if mode == 'PVP': do_das(players[1], pg.K_LEFT, pg.K_RIGHT, pg.K_DOWN)

        # --- Game Logic ---
        for pid, p in players.items():
            if not p.is_local and not p.is_ai: continue
            if p.game_over: continue
            
            # AI Logic
            if p.is_ai:
                if not p.piece.is_fixed:
                    if p.ai_act_timer >= p.ai_act_interval:
                        p.ai_act_timer = 0
                        
                        if p.ai_target_move is None:
                            # 判斷使用哪種 AI
                            if hasattr(p.ai_agent, 'find_best_move'):
                                # Weighted AI
                                p.ai_target_move = p.ai_agent.find_best_move(p.shot, p.piece)
                            else:
                                # Heuristic / Expert AI (使用函式)
                                p.ai_target_move = get_ai_move_heuristic(p.shot, p.piece)
                                
                            if p.ai_target_move is None:
                                p.ai_target_move = (p.piece.x, p.piece.rotation)

                        if p.ai_target_move:
                            tx, tr = p.ai_target_move
                            moved = False
                            
                            if p.piece.rotation != tr:
                                Handler.rotate(p.shot, p.piece)
                                moved = True
                            elif p.piece.x < tx:
                                Handler.moveRight(p.shot, p.piece)
                                moved = True
                            elif p.piece.x > tx:
                                Handler.moveLeft(p.shot, p.piece)
                                moved = True
                            
                            if not moved: 
                                if ai_mode == 'EXPERT':
                                    Handler.instantDrop(p.shot, p.piece)
                                else:
                                    Handler.drop(p.shot, p.piece) 
                    else:
                        p.ai_act_timer += 1

            if not p.is_ai:
                if p.counter >= config.difficulty:
                    Handler.drop(p.shot, p.piece)
                    p.counter = 0
                else:
                    p.counter += 1
                
            if p.piece.is_fixed:
                if p.is_ai: p.ai_target_move = None 
                
                clears, all_clear = Handler.eliminateFilledRows(p.shot, p.piece)
                if clears == 4: p.shot.tetris_timer = 60
                
                atk = 0
                if mode != 'SOLO':
                    is_power = (clears == 4)
                    p.shot.combo_count = p.shot.combo_count + 1 if clears > 0 else 0
                    atk = Handler.calculateAttack(clears, p.shot.combo_count, p.shot.is_b2b, all_clear)
                    p.shot.is_b2b = is_power if is_power else (False if clears > 0 else p.shot.is_b2b)
                
                if atk > 0 and p.shot.pending_garbage > 0:
                    cancel = min(atk, p.shot.pending_garbage)
                    p.shot.pending_garbage -= cancel
                    atk -= cancel
                
                # [問題 3 回答] 垃圾行邏輯
                # 這裡的邏輯是: 當沒有消行 (clears == 0) 時，將垃圾行推入。
                # 這與 AI 的速度無關，而是與「回合(塊)」有關。
                # 如果 AI 變慢，它下塊變慢，垃圾進入的頻率自然變慢，
                # 所以緩衝時間在「邏輯上」是等比例的，不會因為 AI 慢就被瞬間淹沒。
                if clears == 0 and p.shot.pending_garbage > 0:
                    Handler.insertGarbage(p.shot, p.shot.pending_garbage)
                    p.shot.pending_garbage = 0
                    p.shot.shake_timer = 20
                    
                if atk > 0:
                    if mode == 'LAN':
                        net_mgr.total_garbage_sent += atk
                    elif mode in ['PVP', 'PVE']:
                        target_id = 1 if pid == 0 else 0
                        if not players[target_id].game_over:
                            players[target_id].shot.pending_garbage += atk
                        
                p.piece = p.next_piece
                p.next_piece = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
                
                if Handler.isDefeat(p.shot, p.piece):
                    p.game_over = True

        # --- End Conditions ---
        alive_count = sum(1 for p in players.values() if not p.game_over)
        should_check_win = True
        if mode == 'LAN' and len(players) < 2: should_check_win = False
        
        if should_check_win and alive_count == 0:
            if mode != 'SOLO':
                winner_name = "Draw"
                best_score = -1
                for p in players.values():
                    if p.shot.score > best_score:
                        best_score = p.shot.score
                        winner_name = p.name
                return "GAME_OVER", {"winner": winner_name, "score": best_score}
            else:
                return "GAME_OVER", {"winner": "Solo", "score": players[0].shot.score}

        # --- Rendering (修正版) ---
        # [修改 1] 介面佈局邏輯
        screen.fill(config.background_color)
        total_players = len(players)
        
        # 取得第一個玩家的畫面來計算尺寸 (假設大家尺寸一樣)
        temp_surf = draw_player_ui_surface(players[my_id].shot, players[my_id].piece, players[my_id].next_piece, font, players[my_id].name)
        surf_w = temp_surf.get_width()
        surf_h = temp_surf.get_height()
        
        if total_players == 1:
            # SOLO 模式: 絕對置中
            p = players[my_id]
            surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
            
            center_x = (config.width - surf_w) // 2
            center_y = (config.height - surf_h) // 2
            # 確保不會太上面
            if center_y < 20: center_y = 20
            
            screen.blit(surf, (center_x, center_y))
            
            if p.game_over:
                _draw_game_over_overlay(screen, surf, center_x, center_y, font, "DEFEAT")

        elif total_players == 2:
            # 1v1 / 1vAI 模式: 雙人等分置中
            # 計算兩個畫面加上中間間距的總寬度
            gap = 50
            total_w = surf_w * 2 + gap
            start_x = (config.width - total_w) // 2
            y_pos = (config.height - surf_h) // 2
            if y_pos < 20: y_pos = 20
            
            # 排序: 本地玩家(0)在左，對手(1)在右
            sorted_pids = sorted(players.keys())
            
            for i, pid in enumerate(sorted_pids):
                p = players[pid]
                surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
                
                x_pos = start_x + i * (surf_w + gap)
                screen.blit(surf, (x_pos, y_pos))
                
                if p.game_over:
                     _draw_game_over_overlay(screen, surf, x_pos, y_pos, font, "DEFEAT")
                     
        else:
            # LAN 多人模式 (超過2人): 維持縮放邏輯，但置中排列
            sorted_pids = [my_id] + sorted([pid for pid in players if pid != my_id])
            scale = 0.75
            scaled_w = int(surf_w * scale)
            scaled_h = int(surf_h * scale)
            gap = 20
            
            total_w = len(players) * scaled_w + (len(players)-1) * gap
            start_x = (config.width - total_w) // 2
            y_pos = 50
            
            for i, pid in enumerate(sorted_pids):
                p = players[pid]
                surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
                surf = pg.transform.smoothscale(surf, (scaled_w, scaled_h))
                
                x_pos = start_x + i * (scaled_w + gap)
                screen.blit(surf, (x_pos, y_pos))
                
                if p.game_over:
                    _draw_game_over_overlay(screen, surf, x_pos, y_pos, font, "OUT")

        # Tetris timer update
        me = players[my_id]
        if getattr(me.shot, 'tetris_timer', 0) > 0: me.shot.tetris_timer -= 1
        for pid, p in players.items():
            if pid != my_id:
                if getattr(p.shot, 'tetris_timer', 0) > 0: p.shot.tetris_timer -= 1
            
        pg.display.update()
        clock.tick(60)

def _draw_game_over_overlay(screen, surf, x, y, font, text_str):
    """ 輔助函式: 繪製遊戲結束遮罩 """
    s = pg.Surface(surf.get_size(), pg.SRCALPHA)
    s.fill((0,0,0,150))
    screen.blit(s, (x, y))
    txt = font.render(text_str, True, (255, 50, 50))
    screen.blit(txt, txt.get_rect(center=(x + surf.get_width()//2, y + surf.get_height()//2)))