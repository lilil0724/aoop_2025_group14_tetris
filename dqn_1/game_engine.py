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
from menus import pause_menu
from ai_heuristic import get_ai_move_heuristic

# 嘗試匯入 AI
try:
    from ai_player_nn import AIPlayerNN
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: ai_player_nn.py not found. 1vAI mode will be disabled or random.")

# --- 核心遊戲流程 ---
def run_game(screen, clock, font, mode, ai_mode=None, net_mgr=None, sounds=None):
    
    # --- Player Context Helper ---
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
            
            # Controls / Physics
            self.counter = 0
            # Initialize key ticker with all possible control keys
            all_keys = list(settings.KEY_BINDINGS.values())
            self.key_ticker = {k: 0 for k in all_keys}
            
            # AI
            self.ai_nn = None
            self.ai_target_move = None
            self.ai_timer = 0
            self.ai_think_timer = 0

    players = {}
    my_id = 0
    
    # --- Initialization ---
    if mode == 'LAN':
        if net_mgr.is_server:
            my_id = 0
            players[0] = PlayerContext(is_local=True, name="Host (You)")
        else:
            # Wait for ID assignment
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
        players[1] = PlayerContext(is_local=False, is_ai=True, name="AI Bot")
        
        # Load AI
        if ai_mode == 'DQN' and AI_AVAILABLE:
            try:
                players[1].ai_nn = AIPlayerNN(model_path='tetris_dqn_new.pt')
                print("AI: Loaded DQN Model")
            except:
                print("AI: Failed to load model")
                players[1].ai_nn = AIPlayerNN()
        elif ai_mode == 'HEURISTIC':
            print("AI: Heuristic Mode")

    running = True
    paused = False
    send_timer = 0 # [NEW] Network send rate limiter
    
    while running:
        # --- Pause ---
        if paused:
            action = pause_menu(screen)
            if action == "RESUME": paused = False
            elif action == "RESTART": return "RESTART"
            elif action == "MENU": return "MENU"
            clock.tick(30); continue

        # --- Network Sync (LAN) ---
        if mode == 'LAN':
            if not net_mgr.connected: return "MENU"
            
            # 1. Receive Remote Data
            remote_data = net_mgr.get_latest_data()
            for pid, data in remote_data.items():
                if pid == my_id: continue
                
                if pid not in players:
                    players[pid] = PlayerContext(name=f"Player {pid}")
                
                p = players[pid]
                p.shot.status = data['status']
                p.shot.color = data['color']
                p.shot.score = data['score']
                p.shot.line_count = data['lines']
                p.shot.pending_garbage = data.get('pending_garbage', 0) # [NEW] Sync garbage bar
                p.piece.x = data['piece_x']
                p.piece.y = data['piece_y']
                p.piece.shape = data['piece_shape']
                p.piece.rotation = data['piece_rot']
                p.piece.color = data['piece_color']
                p.next_piece.shape = data['next_piece_shape']
                p.next_piece.color = data['next_piece_color']
                p.game_over = data['game_over']
            
            # 2. Incoming Garbage
            diff = net_mgr.get_garbage_diff()
            if diff > 0:
                players[my_id].shot.pending_garbage += diff
                
            # 3. Send Local Data (Rate Limited: Every 3 frames ~ 20 FPS)
            send_timer = (send_timer + 1) % 3
            if send_timer == 0:
                me = players[my_id]
                local_data = {
                    'status': me.shot.status,
                    'color': me.shot.color,
                    'score': me.shot.score,
                    'lines': me.shot.line_count,
                    'pending_garbage': me.shot.pending_garbage, # [NEW] Send my garbage status
                    'piece_x': me.piece.x,
                    'piece_y': me.piece.y,
                    'piece_shape': me.piece.shape,
                    'piece_rot': me.piece.rotation,
                    'piece_color': me.piece.color,
                    'next_piece_shape': me.next_piece.shape,
                    'next_piece_color': me.next_piece.color,
                    'game_over': me.game_over
                }
                net_mgr.send(local_data)

        # --- Event Handling ---
        for event in pg.event.get():
            if event.type == pg.QUIT: pg.quit(); sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE: paused = True
                
                # P1 / Local Controls
                p1 = players[my_id]
                if not p1.game_over:
                    if event.key == settings.KEY_BINDINGS['P1_ROTATE']: Handler.rotate(p1.shot, p1.piece)
                    if event.key == settings.KEY_BINDINGS['P1_DOWN']: p1.key_ticker[settings.KEY_BINDINGS['P1_DOWN']] = 13; Handler.drop(p1.shot, p1.piece)
                    if event.key == settings.KEY_BINDINGS['P1_LEFT']: p1.key_ticker[settings.KEY_BINDINGS['P1_LEFT']] = 13; Handler.moveLeft(p1.shot, p1.piece)
                    if event.key == settings.KEY_BINDINGS['P1_RIGHT']: p1.key_ticker[settings.KEY_BINDINGS['P1_RIGHT']] = 13; Handler.moveRight(p1.shot, p1.piece)
                    if event.key == settings.KEY_BINDINGS['P1_DROP']: Handler.instantDrop(p1.shot, p1.piece)
                
                # P2 Controls (PVP)
                if mode == 'PVP':
                    p2 = players[1]
                    if not p2.game_over:
                        if event.key == settings.KEY_BINDINGS['P2_ROTATE']: Handler.rotate(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_DOWN']: p2.key_ticker[settings.KEY_BINDINGS['P2_DOWN']] = 13; Handler.drop(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_LEFT']: p2.key_ticker[settings.KEY_BINDINGS['P2_LEFT']] = 13; Handler.moveLeft(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_RIGHT']: p2.key_ticker[settings.KEY_BINDINGS['P2_RIGHT']] = 13; Handler.moveRight(p2.shot, p2.piece)
                        if event.key == settings.KEY_BINDINGS['P2_DROP']: Handler.instantDrop(p2.shot, p2.piece)

        # --- DAS Handling ---
        keys = pg.key.get_pressed()
        def do_das(p, k_l, k_r, k_d):
            if p.game_over: return
            if keys[k_l] and p.key_ticker[k_l] == 0: p.key_ticker[k_l] = 6; Handler.moveLeft(p.shot, p.piece)
            if keys[k_r] and p.key_ticker[k_r] == 0: p.key_ticker[k_r] = 6; Handler.moveRight(p.shot, p.piece)
            if keys[k_d] and p.key_ticker[k_d] == 0: p.key_ticker[k_d] = 6; Handler.drop(p.shot, p.piece)
            for k in p.key_ticker:
                if p.key_ticker[k] > 0: p.key_ticker[k] -= 1
        
        do_das(players[my_id], settings.KEY_BINDINGS['P1_LEFT'], settings.KEY_BINDINGS['P1_RIGHT'], settings.KEY_BINDINGS['P1_DOWN'])
        if mode == 'PVP': do_das(players[1], settings.KEY_BINDINGS['P2_LEFT'], settings.KEY_BINDINGS['P2_RIGHT'], settings.KEY_BINDINGS['P2_DOWN'])

        # --- Game Logic (Gravity, AI, Clearing) ---
        for pid, p in players.items():
            if not p.is_local and not p.is_ai: continue # Skip remote players
            if p.game_over: continue
            
            # AI Logic
            ai_dropping = False
            if p.is_ai:
                if p.ai_target_move is None:
                    if p.ai_think_timer < settings.AI_THINKING_DELAY:
                        p.ai_think_timer += 1
                    else:
                        if ai_mode == 'DQN' and p.ai_nn:
                            p.ai_target_move = p.ai_nn.find_best_move(copy.deepcopy(p.shot), copy.deepcopy(p.piece), copy.deepcopy(p.next_piece))
                        elif ai_mode == 'HEURISTIC':
                            p.ai_target_move = get_ai_move_heuristic(p.shot, p.piece)
                        else:
                            p.ai_target_move = (random.randint(0, config.columns-3), random.randint(0, 3))
                        
                        if p.ai_target_move is None: p.ai_target_move = (p.piece.x, p.piece.rotation)
                        p.ai_think_timer = 0
                else:
                    tx, tr = p.ai_target_move
                    aligned = (p.piece.x == tx) and (p.piece.rotation == tr)
                    if p.ai_timer >= settings.AI_MOVE_DELAY:
                        p.ai_timer = 0
                        if p.piece.rotation != tr: Handler.rotate(p.shot, p.piece)
                        elif p.piece.x < tx: Handler.moveRight(p.shot, p.piece)
                        elif p.piece.x > tx: Handler.moveLeft(p.shot, p.piece)
                    else:
                        p.ai_timer += 1
                    
                    if aligned:
                        ai_dropping = True
                        drop_spd = max(2, config.difficulty // 8)
                        if p.counter >= drop_spd: 
                            Handler.drop(p.shot, p.piece)
                            p.counter = 0
                        else: 
                            p.counter += 1

            # Gravity (Only if not AI dropping)
            if not ai_dropping:
                if p.counter >= config.difficulty:
                    Handler.drop(p.shot, p.piece)
                    p.counter = 0
                else:
                    p.counter += 1
                
            # Lock & Clear
            if p.piece.is_fixed:
                if p.is_ai: p.ai_target_move = None # Reset AI
                
                clears, all_clear = Handler.eliminateFilledRows(p.shot, p.piece)
                
                # 播放音效
                if clears > 0 and sounds:
                    if clears == 4 and sounds.get('tetris'):
                        sounds['tetris'].play()
                    elif sounds.get('clear'):
                        sounds['clear'].play()

                if clears == 4: p.shot.tetris_timer = 60
                
                # Attack Calculation
                atk = 0
                if mode != 'SOLO':
                    is_power = (clears == 4)
                    p.shot.combo_count = p.shot.combo_count + 1 if clears > 0 else 0
                    atk = Handler.calculateAttack(clears, p.shot.combo_count, p.shot.is_b2b, all_clear)
                    p.shot.is_b2b = is_power if is_power else (False if clears > 0 else p.shot.is_b2b)
                
                # Garbage Cancellation
                if atk > 0 and p.shot.pending_garbage > 0:
                    cancel = min(atk, p.shot.pending_garbage)
                    p.shot.pending_garbage -= cancel
                    atk -= cancel
                
                # Receive Garbage
                if clears == 0 and p.shot.pending_garbage > 0:
                    Handler.insertGarbage(p.shot, p.shot.pending_garbage)
                    p.shot.pending_garbage = 0
                    p.shot.shake_timer = 20
                    
                # Send Attack
                if atk > 0:
                    if mode == 'LAN':
                        net_mgr.total_garbage_sent += atk
                    elif mode in ['PVP', 'PVE']:
                        target_id = 1 if pid == 0 else 0
                        players[target_id].shot.pending_garbage += atk
                        
                p.piece = p.next_piece
                p.next_piece = pieces.Piece(5, 0, random.choice(list(config.shapes.keys())))
                
                if Handler.isDefeat(p.shot, p.piece):
                    p.game_over = True

        # --- Check End Conditions ---
        alive_count = sum(1 for p in players.values() if not p.game_over)
        
        if mode == 'SOLO':
            if players[0].game_over:
                return "GAME_OVER", {"winner": "Solo", "score": players[0].shot.score, "lines": players[0].shot.line_count}
        else:
            # Multiplayer
            # 只有當所有預期玩家都連線後，才開始檢查勝利條件
            # 簡單判斷：如果目前玩家數 < 2 (且不是 SOLO)，則不判定結束 (等待連線)
            # 但如果是遊戲中途斷線導致剩1人，則應該結束
            # 這裡假設遊戲開始時至少有2人 (LAN模式下等待大廳會確保這點，但目前大廳邏輯較簡單)
            # 修正：LAN 模式下，如果只剩 Host 一人且遊戲剛開始 (其他玩家還沒同步過來)，會誤判
            # 我們可以檢查 net_mgr 是否有足夠的連線
            
            should_check_win = True
            if mode == 'LAN':
                # 如果連線數少於預期 (這裡簡化為至少要有2人資料)，先不判贏
                # 更好的做法是 net_mgr 知道 expected_players
                if len(players) < 2: 
                    should_check_win = False
            
            if should_check_win and alive_count <= 1:
                # If only 1 left, they win. If 0, Draw.
                winner_name = "Draw"
                score = 0
                lines = 0
                for p in players.values():
                    if not p.game_over:
                        winner_name = p.name
                        score = p.shot.score
                        lines = p.shot.line_count
                        break
                
                # Special case: If I am the winner, I might want to keep playing?
                # Usually Tetris Battle ends when last man stands.
                return "GAME_OVER", {"winner": winner_name, "score": score, "lines": lines}

        # --- Rendering ---
        screen.fill(config.background_color)
        
        # --- Rendering ---
        screen.fill(config.background_color)
        
        # Determine Layout based on player count
        total_players = len(players)
        
        if total_players <= 2:
            # 1v1 or Solo: Classic Layout (Left: Me, Right: Opponent)
            me = players[my_id]
            surf_local = draw_player_ui_surface(me.shot, me.piece, me.next_piece, font, me.name)
            
            # Calculate total width to center
            w_local = surf_local.get_width()
            w_op = 0
            gap = 50
            
            others = [p for pid, p in players.items() if pid != my_id]
            if others:
                w_op = w_local # Assuming same size
                total_w = w_local + gap + w_op
            else:
                total_w = w_local
            
            start_x = (config.width - total_w) // 2
            
            # Draw Local
            screen.blit(surf_local, (start_x, 50))
            
            if me.game_over:
                s = pg.Surface(surf_local.get_size(), pg.SRCALPHA)
                s.fill((0,0,0,150))
                screen.blit(s, (start_x, 50))
                txt = font.render("GAME OVER", True, (255, 50, 50))
                screen.blit(txt, txt.get_rect(center=(start_x + surf_local.get_width()//2, 50 + surf_local.get_height()//2)))
            
            if others:
                op = others[0]
                surf_op = draw_player_ui_surface(op.shot, op.piece, op.next_piece, font, op.name)
                x = start_x + w_local + gap
                y = 50
                screen.blit(surf_op, (x, y))
                
                if op.game_over:
                    s = pg.Surface(surf_op.get_size(), pg.SRCALPHA)
                    s.fill((0,0,0,150))
                    screen.blit(s, (x, y))
                    txt = pg.font.SysFont('Arial', 40, bold=True).render("DEFEAT", True, (255, 50, 50))
                    screen.blit(txt, txt.get_rect(center=(x + surf_op.get_width()//2, y + surf_op.get_height()//2)))
        
        else:
            # Multiplayer (>2): Grid Layout (All Equal Size)
            # Sort players: Me first, then by ID
            sorted_pids = [my_id] + sorted([pid for pid in players if pid != my_id])
            
            # Calculate scale to fit width
            # Original UI Width ~522
            # Window Width = 1600
            # Max 4 players side-by-side: 1600 / 4 = 400 per player
            # Scale = 400 / 522 ~= 0.76
            
            scale = 0.75
            base_w = 522 # Approx
            scaled_w = int(base_w * scale)
            scaled_h = int((config.rows * config.grid + 50) * scale)
            
            gap_x = 20
            total_w = len(sorted_pids) * scaled_w + (len(sorted_pids) - 1) * gap_x
            start_x = (config.width - total_w) // 2
            start_y = 50
            
            for i, pid in enumerate(sorted_pids):
                p = players[pid]
                surf = draw_player_ui_surface(p.shot, p.piece, p.next_piece, font, p.name)
                surf = pg.transform.smoothscale(surf, (scaled_w, scaled_h))
                
                x = start_x + i * (scaled_w + gap_x)
                y = start_y
                
                screen.blit(surf, (x, y))
                
                if p.game_over:
                    s = pg.Surface((scaled_w, scaled_h), pg.SRCALPHA)
                    s.fill((0,0,0,150))
                    screen.blit(s, (x, y))
                    txt = pg.font.SysFont('Arial', 30, bold=True).render("OUT", True, (255, 50, 50))
                    screen.blit(txt, txt.get_rect(center=(x + scaled_w//2, y + scaled_h//2)))

        # Update Display
        me = players[my_id] # Re-fetch for timer update
        if getattr(me.shot, 'tetris_timer', 0) > 0: me.shot.tetris_timer -= 1
        for pid, p in players.items():
            if pid != my_id:
                if getattr(p.shot, 'tetris_timer', 0) > 0: p.shot.tetris_timer -= 1
            
        pg.display.update()
        clock.tick(60)
