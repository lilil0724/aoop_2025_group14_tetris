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
from ui import draw_player_ui_surface, Button
from ai_heuristic import get_ai_move_heuristic
from ai_weighted import WeightedAI
from menus import pause_menu

# 7-Bag 生成器
def get_new_bag():
    bag = list(config.shapes.keys())
    random.shuffle(bag)
    return bag

def run_game(screen, clock, font, mode, ai_mode=None, net_mgr=None, sounds=None):
    
    class PlayerContext:
        def __init__(self, is_local=False, is_ai=False, name="Player"):
            self.shot = shots.Shot()
            self.shot.tetris_timer = 0
            
            # 7-Bag 初始化
            self.bag = get_new_bag() 
            self.piece = pieces.Piece(5, 0, self.bag.pop())
            if not self.bag: self.bag = get_new_bag()
            self.next_piece = pieces.Piece(5, 0, self.bag.pop())
            
            self.game_over = False
            self.is_local = is_local
            self.is_ai = is_ai
            self.name = name
            
            self.counter = 0
            # 定義所有可能用到的按鍵 ticker
            self.key_ticker = {k: 0 for k in settings.KEY_BINDINGS.values()}
            self.key_ticker[pg.K_SPACE] = 0 # 額外支援 Space
            
            self.ai_target_move = None 
            self.ai_act_timer = 0
            self.ai_act_interval = 5 
            self.ai_agent = None 

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
        speed_map = { 1: 15, 2: 8, 3: 3, 4: 0 }
        selected_speed_delay = speed_map.get(settings.AI_SPEED_LEVEL, 8)
        
        if ai_mode == 'WEIGHT':
            players[1] = PlayerContext(is_local=False, is_ai=True, name="Weighted AI")
            players[1].ai_agent = WeightedAI() 
            players[1].ai_act_interval = selected_speed_delay
            print(f"Loaded Weighted AI. Speed Level: {settings.AI_SPEED_LEVEL}")
        elif ai_mode == 'EXPERT': 
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
    surrender_btn = Button(config.width // 2 - 100, config.height - 80, 200, 50, "Surrender", "SURRENDER", color=(200, 50, 50))
    surrendered = False
    
    while running:
        # Check if paused by host (Guest side)
        if mode == 'LAN' and net_mgr and not net_mgr.is_server and net_mgr.paused:
            paused = True

        if paused:
            action = pause_menu(screen, net_mgr=net_mgr, mode=mode)
            if action == "RESUME": paused = False
            elif action == "RESTART": return "RESTART"
            elif action == "MENU": return "MENU"
            clock.tick(30); continue

        # --- Network Sync (LAN) ---
        if mode == 'LAN':
            if not net_mgr.connected: return "MENU"
            
            # Check for restart signal from Host
            if net_mgr.restart_requested:
                return "RESTART"
                
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
            
            if mode == 'PVE' and players[0].game_over:
                if surrender_btn.is_clicked(event):
                    surrendered = True

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    if mode == 'LAN':
                        # Only Host can pause in LAN
                        if net_mgr and net_mgr.is_server:
                            paused = True
                    else:
                        paused = True
                
                # --- 模式區分 ---
                if mode == 'PVP':
                    # P1: 使用設定鍵位
                    p1 = players[0]
                    if not p1.game_over:
                        if event.key == settings.KEY_BINDINGS['P1_ROTATE']: Handler.rotate(p1.shot, p1.piece)
                        if event.key == settings.KEY_BINDINGS['P1_ROTATE_CCW']: Handler.rotateCCW(p1.shot, p1.piece)
                        if event.key == settings.KEY_BINDINGS['P1_DOWN']: p1.key_ticker[event.key] = 13; Handler.drop(p1.shot, p1.piece)
                        if event.key == settings.KEY_BINDINGS['P1_LEFT']: p1.key_ticker[event.key] = 13; Handler.moveLeft(p1.shot, p1.piece)
                        if event.key == settings.KEY_BINDINGS['P1_RIGHT']: p1.key_ticker[event.key] = 13; Handler.moveRight(p1.shot, p1.piece)
                        if event.key == settings.KEY_BINDINGS['P1_DROP']: Handler.instantDrop(p1.shot, p1.piece)
                    
                    # P2: 使用設定鍵位
                    p2 = players[1]
                    if not p2.game_over:
                        if event.key == settings.KEY_BINDINGS['P2_ROTATE']: Handler.rotate(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_ROTATE_CCW']: Handler.rotateCCW(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_DOWN']: p2.key_ticker[event.key] = 13; Handler.drop(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_LEFT']: p2.key_ticker[event.key] = 13; Handler.moveLeft(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_RIGHT']: p2.key_ticker[event.key] = 13; Handler.moveRight(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_DROP']: Handler.instantDrop(p2.shot, p2.piece)
                
                else:
                    # SOLO, PVE, LAN -> 統一使用 P1 鍵位，並額外支援 Space
                    p_local = players[my_id]
                    if not p_local.game_over:
                        if event.key == settings.KEY_BINDINGS['P1_ROTATE']: Handler.rotate(p_local.shot, p_local.piece)
                        if event.key == settings.KEY_BINDINGS['P1_ROTATE_CCW']: Handler.rotateCCW(p_local.shot, p_local.piece)
                        if event.key == settings.KEY_BINDINGS['P1_DOWN']: p_local.key_ticker[event.key] = 13; Handler.drop(p_local.shot, p_local.piece)
                        if event.key == settings.KEY_BINDINGS['P1_LEFT']: p_local.key_ticker[event.key] = 13; Handler.moveLeft(p_local.shot, p_local.piece)
                        if event.key == settings.KEY_BINDINGS['P1_RIGHT']: p_local.key_ticker[event.key] = 13; Handler.moveRight(p_local.shot, p_local.piece)
                        if event.key == settings.KEY_BINDINGS['P1_DROP'] or event.key == pg.K_SPACE: Handler.instantDrop(p_local.shot, p_local.piece)

        # --- DAS ---
        keys = pg.key.get_pressed()
        def do_das(p, k_l, k_r, k_d):
            if p.game_over: return
            if keys[k_l] and p.key_ticker.get(k_l, 0) == 0: p.key_ticker[k_l] = 6; Handler.moveLeft(p.shot, p.piece)
            if keys[k_r] and p.key_ticker.get(k_r, 0) == 0: p.key_ticker[k_r] = 6; Handler.moveRight(p.shot, p.piece)
            if keys[k_d] and p.key_ticker.get(k_d, 0) == 0: p.key_ticker[k_d] = 6; Handler.drop(p.shot, p.piece)
            for k in p.key_ticker:
                if p.key_ticker[k] > 0: p.key_ticker[k] -= 1
        
        if mode == 'PVP':
            do_das(players[0], settings.KEY_BINDINGS['P1_LEFT'], settings.KEY_BINDINGS['P1_RIGHT'], settings.KEY_BINDINGS['P1_DOWN'])
            do_das(players[1], settings.KEY_BINDINGS['P2_LEFT'], settings.KEY_BINDINGS['P2_RIGHT'], settings.KEY_BINDINGS['P2_DOWN'])
        else:
            do_das(players[my_id], settings.KEY_BINDINGS['P1_LEFT'], settings.KEY_BINDINGS['P1_RIGHT'], settings.KEY_BINDINGS['P1_DOWN'])

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
                            if hasattr(p.ai_agent, 'find_best_move'):
                                p.ai_target_move = p.ai_agent.find_best_move(p.shot, p.piece)
                            else:
                                p.ai_target_move = get_ai_move_heuristic(p.shot, p.piece)
                            if p.ai_target_move is None:
                                p.ai_target_move = (p.piece.x, p.piece.rotation)
                        if p.ai_target_move:
                            tx, tr = p.ai_target_move
                            moved = False
                            if p.piece.rotation != tr: Handler.rotate(p.shot, p.piece); moved = True
                            elif p.piece.x < tx: Handler.moveRight(p.shot, p.piece); moved = True
                            elif p.piece.x > tx: Handler.moveLeft(p.shot, p.piece); moved = True
                            if not moved: 
                                if ai_mode == 'EXPERT': Handler.instantDrop(p.shot, p.piece)
                                else: Handler.drop(p.shot, p.piece) 
                    else: p.ai_act_timer += 1

            if not p.is_ai:
                if p.counter >= config.difficulty:
                    Handler.drop(p.shot, p.piece)
                    p.counter = 0
                else: p.counter += 1
                
            if p.piece.is_fixed:
                if p.is_ai: p.ai_target_move = None 
                
                clears, all_clear = Handler.eliminateFilledRows(p.shot, p.piece)

                if p.is_local and sounds:
                    if clears == 4 and 'tetris' in sounds:
                        sounds['tetris'].play()
                    elif clears > 0 and 'clear' in sounds:
                        sounds['clear'].play()
                
                if clears == 4: p.shot.tetris_timer = 60
                if all_clear: p.shot.all_clear_timer = 60 
                
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
                if not p.bag: p.bag = get_new_bag()
                p.next_piece = pieces.Piece(5, 0, p.bag.pop())
                
                if Handler.isDefeat(p.shot, p.piece): p.game_over = True

        # --- End Conditions ---
        alive_count = sum(1 for p in players.values() if not p.game_over)
        should_check_win = True
        if mode == 'LAN' and len(players) < 2: should_check_win = False
        
        game_is_over = False
        if surrendered:
            game_is_over = True
        elif should_check_win:
            if mode == 'PVE':
                # PVE: 等待 AI 結束，除非玩家投降
                if alive_count == 0: game_is_over = True
            elif len(players) > 1:
                # Multiplayer (PVP/LAN): End only when EVERYONE is dead (Score Attack)
                if alive_count == 0: game_is_over = True
            else:
                # Solo: End if 0 survivors
                if alive_count == 0: game_is_over = True
        
        if game_is_over:
            # Send final state immediately to ensure others know I'm dead/game over
            if mode == 'LAN' and net_mgr:
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
                # Give a tiny bit of time for the packet to go out
                time.sleep(0.1)

            results = []
            
            is_draw = False
            winner_p = None
            
            if surrendered:
                # PVE Surrender: Player 0 loses, Player 1 wins
                if 1 in players: winner_p = players[1]
            else:
                # Sort by score to determine winner
                sorted_players = sorted(players.values(), key=lambda p: p.shot.score, reverse=True)
                
                # Check for Draw (if top 2 have same score)
                if len(sorted_players) > 1 and sorted_players[0].shot.score == sorted_players[1].shot.score:
                    is_draw = True
                else:
                    winner_p = sorted_players[0]

            for pid in sorted(players.keys()):
                p = players[pid]
                results.append({
                    "name": p.name,
                    "score": p.shot.score,
                    "lines": p.shot.line_count,
                    "is_winner": (p == winner_p),
                    "is_draw": is_draw,
                    "is_local": p.is_local
                })
            return "GAME_OVER", results

        # --- Rendering ---
        screen.fill(config.background_color)
        total_players = len(players)
        temp_surf = draw_player_ui_surface(players[my_id].shot, players[my_id].piece, players[my_id].next_piece, font, players[my_id].name)
        surf_w, surf_h = temp_surf.get_width(), temp_surf.get_height()
        
        if total_players == 1:
            p = players[my_id]
            surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
            center_x = (config.width - surf_w) // 2
            center_y = max(20, (config.height - surf_h) // 2)
            screen.blit(surf, (center_x, center_y))
            if p.game_over: _draw_game_over_overlay(screen, surf, center_x, center_y, font, "DEFEAT")

        elif total_players == 2:
            gap = 50
            total_w = surf_w * 2 + gap
            start_x = (config.width - total_w) // 2
            y_pos = max(20, (config.height - surf_h) // 2)
            for i, pid in enumerate(sorted(players.keys())):
                p = players[pid]
                surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
                x_pos = start_x + i * (surf_w + gap)
                screen.blit(surf, (x_pos, y_pos))
                if p.game_over: _draw_game_over_overlay(screen, surf, x_pos, y_pos, font, "DEFEAT")
            
            # PVE 模式下，如果玩家輸了但 AI 還在跑，顯示投降按鈕
            if mode == 'PVE' and players[0].game_over and not game_is_over:
                surrender_btn.draw(screen)
                     
        else:
            # 3P / 4P Layout
            sorted_pids = [my_id] + sorted([pid for pid in players if pid != my_id])
            
            if total_players == 3:
                # Try 3 columns first
                gap = 20
                scale_w = (config.width - 4 * gap) / 3
                scale = min(1.0, scale_w / surf_w)
                
                # If scale is too small (e.g. < 0.6), try 2 rows (2 top, 1 bottom)
                if scale < 0.6:
                     # 2 Rows layout
                     scale_w = (config.width - 3 * gap) / 2
                     scale_h = (config.height - 3 * gap) / 2
                     scale = min(1.0, min(scale_w / surf_w, scale_h / surf_h))
                     
                     scaled_w, scaled_h = int(surf_w * scale), int(surf_h * scale)
                     
                     # Row 1 (2 players)
                     start_x_r1 = (config.width - (2 * scaled_w + gap)) // 2
                     y_r1 = gap
                     
                     # Row 2 (1 player)
                     start_x_r2 = (config.width - scaled_w) // 2
                     y_r2 = y_r1 + scaled_h + gap
                     
                     for i, pid in enumerate(sorted_pids):
                        p = players[pid]
                        surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
                        surf = pg.transform.smoothscale(surf, (scaled_w, scaled_h))
                        
                        if i < 2:
                            x_pos = start_x_r1 + i * (scaled_w + gap)
                            y_pos = y_r1
                        else:
                            x_pos = start_x_r2
                            y_pos = y_r2
                            
                        screen.blit(surf, (x_pos, y_pos))
                        if p.game_over: _draw_game_over_overlay(screen, surf, x_pos, y_pos, font, "OUT")
                else:
                    # 3 Columns layout
                    scaled_w, scaled_h = int(surf_w * scale), int(surf_h * scale)
                    total_w = 3 * scaled_w + 2 * gap
                    start_x = (config.width - total_w) // 2
                    y_pos = (config.height - scaled_h) // 2
                    
                    for i, pid in enumerate(sorted_pids):
                        p = players[pid]
                        surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
                        surf = pg.transform.smoothscale(surf, (scaled_w, scaled_h))
                        x_pos = start_x + i * (scaled_w + gap)
                        screen.blit(surf, (x_pos, y_pos))
                        if p.game_over: _draw_game_over_overlay(screen, surf, x_pos, y_pos, font, "OUT")
            
            else:
                # 4 Players (2x2 Grid)
                gap = 20
                scale_w = (config.width - 3 * gap) / 2
                scale_h = (config.height - 3 * gap) / 2
                scale = min(1.0, min(scale_w / surf_w, scale_h / surf_h))
                
                scaled_w, scaled_h = int(surf_w * scale), int(surf_h * scale)
                
                start_x = (config.width - (2 * scaled_w + gap)) // 2
                start_y = (config.height - (2 * scaled_h + gap)) // 2
                
                for i, pid in enumerate(sorted_pids):
                    p = players[pid]
                    surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
                    surf = pg.transform.smoothscale(surf, (scaled_w, scaled_h))
                    
                    col = i % 2
                    row = i // 2
                    
                    x_pos = start_x + col * (scaled_w + gap)
                    y_pos = start_y + row * (scaled_h + gap)
                    
                    screen.blit(surf, (x_pos, y_pos))
                    if p.game_over: _draw_game_over_overlay(screen, surf, x_pos, y_pos, font, "OUT")

        me = players[my_id]
        if getattr(me.shot, 'tetris_timer', 0) > 0: me.shot.tetris_timer -= 1
        if getattr(me.shot, 'all_clear_timer', 0) > 0: me.shot.all_clear_timer -= 1
        for pid, p in players.items():
            if pid != my_id:
                if getattr(p.shot, 'tetris_timer', 0) > 0: p.shot.tetris_timer -= 1
                if getattr(p.shot, 'all_clear_timer', 0) > 0: p.shot.all_clear_timer -= 1
            
        pg.display.update()
        clock.tick(60)

def _draw_game_over_overlay(screen, surf, x, y, font, text_str):
    s = pg.Surface(surf.get_size(), pg.SRCALPHA)
    s.fill((0,0,0,150))
    screen.blit(s, (x, y))
    txt = font.render(text_str, True, (255, 50, 50))
    screen.blit(txt, txt.get_rect(center=(x + surf.get_width()//2, y + surf.get_height()//2)))