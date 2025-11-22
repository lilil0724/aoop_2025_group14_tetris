import pygame as pg
import pieces
import shots
import config
import Handler
import random
import copy
import torch
import numpy as np
from ai_model import TetrisTransformer
# 也要確保能 import tetris_env，如果不行的話，可以把 tetris_env.py 裡的 helper function 複製過來
import tetris_env 

DEBUG = False

init_start = (5, 0) # 放置新方塊的位置 (邏輯座標)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model = TetrisTransformer().to(device)
try:
    ai_model.load_state_dict(torch.load("tetris_transformer.pth", map_location=device))
    ai_model.eval()
    print("AI 模型載入成功！")
except:
    print("警告：找不到模型檔案，AI 將會隨機行動。")

# --- 新增 helper function ---
def get_ai_move_from_model(shot, piece, model, device):
    """使用 Transformer 模型來決定最佳移動"""
    # 借用 TetrisEnv 的邏輯來生成所有可能的下一步
    # 這裡我們建立一個 "虛擬環境" 來模擬
    temp_env = tetris_env.TetrisEnv()
    
    # 將當前的遊戲狀態注入虛擬環境
    # 注意: shot.status 必須是 numpy array 或 list
    temp_env.board = np.array(shot.status) 
    temp_env.current_piece = copy.deepcopy(piece) # 確保不影響原本的 piece
    
    # 取得所有可能的下一步 (States)
    possible_states = temp_env.get_possible_next_states()
    
    if not possible_states:
        return None # 無路可走
        
    moves = list(possible_states.keys()) # [(x, rot), ...]
    states = list(possible_states.values()) # [numpy_array, ...]
    
    # 轉成 Tensor 並丟進 GPU
    # states 是一個 list of numpy arrays
    state_batch = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    
    with torch.no_grad(): # 推論模式，不需要計算梯度
        scores = model(state_batch) # AI 給每個盤面打分數
        best_idx = scores.argmax().item() # 選分數最高的那個 index
        
    return moves[best_idx] # 回傳最佳的 (x, rotation)


def getRandomPiece():
    shape = random.choice(list(config.shapes.keys()))
    piece = pieces.Piece(*init_start, shape)
    return piece

def draw_grid(surface, offset_x):
    """半透明灰色格線"""
    grid_surface = pg.Surface(
        (config.columns * config.grid, config.rows * config.grid), pg.SRCALPHA
    )
    color = (150, 150, 150, 60)
    for y in range(config.rows):
        pg.draw.line(grid_surface, color, (0, y * config.grid), (config.columns * config.grid, y * config.grid))
    for x in range(config.columns):
        pg.draw.line(grid_surface, color, (x * config.grid, 0), (x * config.grid, config.rows * config.grid))
    surface.blit(grid_surface, (offset_x, 0))


# (修改) 增加 garbage_bar_pos 參數
def draw_player_ui(screen, shot, piece, next_piece, font, 
                   offset_x, score_pos, line_pos, next_piece_pos, 
                   garbage_bar_pos): 
    
    # 1. 更新 shot 狀態 (邏輯)
    for y in range(config.rows):
        for x in range(config.columns):
            if shot.status[y][x] != 2:
                shot.status[y][x] = 0

    if not piece.is_fixed:
        for y, x in Handler.getCellsAbsolutePosition(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                shot.color[y][x] = piece.color
                shot.status[y][x] = 1

    # 2. 繪製遊戲板 (繪圖)
    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            if shot.status[y][x] == 0:
                draw_color = (0, 0, 0)
            else:
                draw_color = color
            pg.draw.rect(screen, draw_color, (
                offset_x + x * config.grid,
                y * config.grid,
                config.grid,
                config.grid
            ))

    # 3. 繪製分數
    textsurface = font.render('Score: {}'.format(
        shot.score), False, (255, 255, 255))
    screen.blit(textsurface, score_pos)

    # 4. 繪製行數
    textsurface = font.render('Line: {}'.format(
        shot.line_count), False, (255, 255, 255))
    screen.blit(textsurface, line_pos)

    # 5. 繪製下一個方塊 (背景)
    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (
                next_piece_pos[0] + x * config.grid,
                next_piece_pos[1] + y * config.grid,
                config.grid,
                config.grid
            ))

    # 6. 繪製下一個方塊 (方塊本身)
    for y, x in next_piece.getCells():
        color = next_piece.color
        pg.draw.rect(screen, color, (
            next_piece_pos[0] + x * config.grid,
            next_piece_pos[1] + y * config.grid,
            config.grid,
            config.grid
        ))
    
    # --- (新增) 7. 繪製垃圾行提示條 ---
    if shot.pending_garbage > 0:
        bar_max_height = config.height * 0.9 # 條的總高度
        bar_y_start = config.height * 0.05
        
        # 視覺上限，例如最多顯示 12 行
        pending_visual = min(shot.pending_garbage, 12) 
        bar_fill_ratio = pending_visual / 12.0
        bar_height = bar_max_height * bar_fill_ratio
        
        bar_x = garbage_bar_pos[0]
        bar_y_fill = (bar_y_start + bar_max_height) - bar_height # 從底部往上填滿
        
        # 灰色底條
        pg.draw.rect(screen, (80, 80, 80), (
            bar_x, bar_y_start,
            config.GARBAGE_BAR_WIDTH, bar_max_height
        ))
        # 紅色填充條
        pg.draw.rect(screen, (255, 50, 50), (
            bar_x, bar_y_fill,
            config.GARBAGE_BAR_WIDTH, bar_height
        ))
    # --- (新增結束) ---

    # 8. 繪製網格
    draw_grid(screen, offset_x)

def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris 1v1: Human vs Transformer AI")

    # --- 初始化 AI 模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model = TetrisTransformer(config.rows, config.columns).to(device)
    try:
        ai_model.load_state_dict(torch.load("tetris_transformer.pth", map_location=device))
        ai_model.eval() # 設定為評估模式
        print(f"成功載入 AI 模型！使用裝置: {device}")
    except Exception as e:
        print(f"警告：找不到模型檔案 ({e})，AI 將無法運作或隨機行動。")
        # 如果你想在沒模型時 fallback 回原本的 ai_player，可以在這裡設定 flag

    # --- P1 遊戲狀態 ---
    shot1 = shots.Shot()
    piece1 = getRandomPiece()
    next_piece1 = getRandomPiece()
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    game_over1 = False

    # --- P2 遊戲狀態 (AI) ---
    shot2 = shots.Shot()
    piece2 = getRandomPiece()
    next_piece2 = getRandomPiece()
    # counter2 = 0 # AI 不需要手動下落計時器，因為它是瞬間移動的
    game_over2 = False
    
    # --- AI 開局第一手 ---
    if not game_over2:
        # 使用新的 AI 模型函式
        best_move = get_ai_move_from_model(shot2, piece2, ai_model, device)
        
        if best_move:
            piece2.x, piece2.rotation = best_move
            Handler.instantDrop(shot2, piece2)
        else:
            # 如果 AI 找不到路，就隨便放 (通常這時候已經死了)
            Handler.instantDrop(shot2, piece2)
    
    run = True
    while run:
        # --- 自動下落計時器 ---
        if not DEBUG:
            if not game_over1:
                if counter1 == config.difficulty:
                    Handler.drop(shot1, piece1)
                    counter1 = 0
                else:
                    counter1 += 1
            # P2 (AI) 不需要自動下落，因為它是邏輯觸發式的

        # --- (新增) 垃圾行插入計時器 ---
        # P1
        if not game_over1 and shot1.pending_garbage > 0:
            shot1.garbage_insert_timer += 1
            if shot1.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                lines_to_add = min(shot1.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                Handler.insertGarbage(shot1, lines_to_add)
                shot1.pending_garbage -= lines_to_add
                shot1.garbage_insert_timer = 0
                if Handler.isDefeat(shot1, piece1):
                    game_over1 = True
                    print("Player 1 Game Over!! (Killed by garbage)")
        # P2
        if not game_over2 and shot2.pending_garbage > 0:
            shot2.garbage_insert_timer += 1
            if shot2.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                lines_to_add = min(shot2.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                Handler.insertGarbage(shot2, lines_to_add)
                shot2.pending_garbage -= lines_to_add
                shot2.garbage_insert_timer = 0
                if Handler.isDefeat(shot2, piece2):
                    game_over2 = True
                    print("Player 2 Game Over!! (Killed by garbage)")
        # --- (新增結束) ---


        # --- 事件處理 (單次按鍵) ---
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    run = False
                
                # P1 (WASD + LShift)
                if not game_over1:
                    if event.key == pg.K_w: Handler.rotate(shot1, piece1)
                    if event.key == pg.K_s:
                        key_ticker1[pg.K_s] = 13
                        Handler.drop(shot1, piece1)
                    if event.key == pg.K_a:
                        key_ticker1[pg.K_a] = 13
                        Handler.moveLeft(shot1, piece1)
                    if event.key == pg.K_d:
                        key_ticker1[pg.K_d] = 13
                        Handler.moveRight(shot1, piece1)
                    if event.key == pg.K_LSHIFT: # P1 瞬降
                        Handler.instantDrop(shot1, piece1)

        # --- 按鍵長按處理 ---
        keys = pg.key.get_pressed()

        # P1 手動控制
        if not game_over1:
            if keys[pg.K_a] and key_ticker1[pg.K_a] == 0:
                key_ticker1[pg.K_a] = 6
                Handler.moveLeft(shot1, piece1)
            if keys[pg.K_d] and key_ticker1[pg.K_d] == 0:
                key_ticker1[pg.K_d] = 6
                Handler.moveRight(shot1, piece1)
            if keys[pg.K_s] and key_ticker1[pg.K_s] == 0:
                key_ticker1[pg.K_s] = 6
                Handler.drop(shot1, piece1)

        for k in key_ticker1:
            if key_ticker1[k] > 0: key_ticker1[k] -= 1

        # --- P1 邏輯：方塊固定與攻擊流程 ---
        if not game_over1:
            if piece1.is_fixed:
                (clears, all_clear) = Handler.eliminateFilledRows(shot1, piece1)
                is_power_move = (clears == 4)
                if clears > 0:
                    shot1.combo_count += 1
                else:
                    shot1.combo_count = 0
                atk1 = Handler.calculateAttack(clears, shot1.combo_count, shot1.is_b2b, all_clear)
                if is_power_move:
                    shot1.is_b2b = True
                elif clears > 0:
                    shot1.is_b2b = False
                if atk1 > 0:
                    cancel_amt = min(atk1, shot1.pending_garbage)
                    shot1.pending_garbage -= cancel_amt
                    atk1 -= cancel_amt
                if atk1 > 0:
                    shot2.pending_garbage += atk1
                piece1, next_piece1 = next_piece1, getRandomPiece()
                if Handler.isDefeat(shot1, piece1):
                    game_over1 = True
                    print("Player 1 Game Over!!")
                    print("P1 Score:", shot1.score, "| P1 Lines:", shot1.line_count)

        # --- P2 邏輯：AI 決策、放置與攻擊流程 ---
        if not game_over2:
            # P2 輪到它了 (piece2.is_fixed 為 True 代表上一個方塊剛放好，輪到這回合)
            if piece2.is_fixed: 

                # 1. 先結算上一個方塊的成績 (消行、攻擊)
                (clears, all_clear) = Handler.eliminateFilledRows(shot2, piece2)
                is_power_move = (clears == 4)
                if clears > 0:
                    shot2.combo_count += 1
                else:
                    shot2.combo_count = 0

                atk2 = Handler.calculateAttack(clears, shot2.combo_count, shot2.is_b2b, all_clear)

                if is_power_move:
                    shot2.is_b2b = True
                elif clears > 0:
                    shot2.is_b2b = False

                if atk2 > 0:
                    cancel_amt = min(atk2, shot2.pending_garbage)
                    shot2.pending_garbage -= cancel_amt
                    atk2 -= cancel_amt
                if atk2 > 0:
                    shot1.pending_garbage += atk2

                # 2. 生成新方塊
                piece2, next_piece2 = next_piece2, getRandomPiece()

                # 3. 檢查是否 Spawn Die
                if Handler.isDefeat(shot2, piece2):
                    game_over2 = True
                    print("Player 2 Game Over!!")
                    print("P2 Score:", shot2.score, "| P2 Lines:", shot2.line_count)

                # 4. (AI 核心) 新方塊生成後，AI 立即思考下一步
                if not game_over2:
                    # 改用新的 Transformer 模型
                    best_move = get_ai_move_from_model(shot2, piece2, ai_model, device)

                    if best_move:
                        (best_x, best_rotation) = best_move
                        piece2.x = best_x
                        piece2.rotation = best_rotation
                        
                        # AI 瞬間放置
                        Handler.instantDrop(shot2, piece2)
                    else:
                        # 無路可走，隨便放等死
                        Handler.instantDrop(shot2, piece2)

        # --- 遊戲結束判斷 ---
        if game_over1 and game_over2:
            run = False
        
        # --- 繪製畫面 ---
        screen.fill(config.background_color)

        if not game_over1:
            draw_player_ui(screen, shot1, piece1, next_piece1, myfont,
                           config.P1_OFFSET_X, config.P1_SCORE_POS, 
                           config.P1_LINE_POS, config.P1_NEXT_PIECE_POS,
                           config.P1_GARBAGE_BAR_POS)
        
        if not game_over2:
            draw_player_ui(screen, shot2, piece2, next_piece2, myfont,
                           config.P2_OFFSET_X, config.P2_SCORE_POS,
                           config.P2_LINE_POS, config.P2_NEXT_PIECE_POS,
                           config.P2_GARBAGE_BAR_POS)

        pg.display.update()
        fpsClock.tick(config.fps)
    
    print("----- Final Result -----")
    print(f"P1 Score: {shot1.score} | P1 Lines: {shot1.line_count}")
    print(f"P2 Score: {shot2.score} | P2 Lines: {shot2.line_count}")
    
    if game_over1 and not game_over2:
        print("Player 2 Wins!")
    elif game_over2 and not game_over1:
        print("Player 1 Wins!")
    else:
        print("It's a Draw!")
        
    pg.quit()



"""
def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris 1v1")

    # --- P1 遊戲狀態 ---
    shot1 = shots.Shot()
    piece1 = getRandomPiece()
    next_piece1 = getRandomPiece()
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    game_over1 = False

    # --- P2 遊戲狀態 ---
    shot2 = shots.Shot()
    piece2 = getRandomPiece()
    next_piece2 = getRandomPiece()
    counter2 = 0
    key_ticker2 = {pg.K_LEFT: 0, pg.K_DOWN: 0, pg.K_RIGHT: 0}
    game_over2 = False
    if not game_over2:
        # 1. AI 決定最佳移動
        best_move = ai_player.find_best_move(copy.deepcopy(shot2), copy.deepcopy(piece2))
        
        # 2. 應用並立即放置
        if best_move:
            piece2.x, piece2.rotation = best_move
            Handler.instantDrop(shot2, piece2)
        else:
            # 如果 AI 找不到（例如開局就死了），隨便放
            Handler.instantDrop(shot2, piece2)
    
    run = True
    while run:
        # --- 自動下落計時器 ---
        if not DEBUG:
            if not game_over1:
                if counter1 == config.difficulty:
                    Handler.drop(shot1, piece1)
                    counter1 = 0
                else:
                    counter1 += 1
#            if not game_over2:
#                if counter2 == config.difficulty:
#                    Handler.drop(shot2, piece2)
#                    counter2 = 0
#                else:
#                    counter2 += 1

        # --- (新增) 垃圾行插入計時器 ---
        # P1
        if not game_over1 and shot1.pending_garbage > 0:
            shot1.garbage_insert_timer += 1
            if shot1.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                lines_to_add = min(shot1.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                Handler.insertGarbage(shot1, lines_to_add)
                shot1.pending_garbage -= lines_to_add
                shot1.garbage_insert_timer = 0
                if Handler.isDefeat(shot1, piece1):
                    game_over1 = True
                    print("Player 1 Game Over!! (Killed by garbage)")
        # P2
        if not game_over2 and shot2.pending_garbage > 0:
            shot2.garbage_insert_timer += 1
            if shot2.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                lines_to_add = min(shot2.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                Handler.insertGarbage(shot2, lines_to_add)
                shot2.pending_garbage -= lines_to_add
                shot2.garbage_insert_timer = 0
                if Handler.isDefeat(shot2, piece2):
                    game_over2 = True
                    print("Player 2 Game Over!! (Killed by garbage)")
        # --- (新增結束) ---


        # --- 事件處理 (單次按鍵) ---
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    run = False
                
                # P1 (WASD + LShift)
                if not game_over1:
                    if event.key == pg.K_w: Handler.rotate(shot1, piece1)
                    if event.key == pg.K_s:
                        key_ticker1[pg.K_s] = 13
                        Handler.drop(shot1, piece1)
                    if event.key == pg.K_a:
                        key_ticker1[pg.K_a] = 13
                        Handler.moveLeft(shot1, piece1)
                    if event.key == pg.K_d:
                        key_ticker1[pg.K_d] = 13
                        Handler.moveRight(shot1, piece1)
                    # --- (修改 1) ---
                    if event.key == pg.K_LSHIFT: # P1 瞬降 (原為 SPACE)
                        Handler.instantDrop(shot1, piece1)

#                # P2 (方向鍵 + Space)
#                if not game_over2:
#                    if event.key == pg.K_UP: Handler.rotate(shot2, piece2)
#                    if event.key == pg.K_DOWN:
#                        key_ticker2[pg.K_DOWN] = 13
#                        Handler.drop(shot2, piece2)
#                    if event.key == pg.K_LEFT:
#                        key_ticker2[pg.K_LEFT] = 13
#                        Handler.moveLeft(shot2, piece2)
#                    if event.key == pg.K_RIGHT:
#                        key_ticker2[pg.K_RIGHT] = 13
#                        Handler.moveRight(shot2, piece2)
#                    # --- (修改 2) ---
#                    if event.key == pg.K_RSHIFT: # P2 瞬降 (使用 Right Shift)
#                        Handler.instantDrop(shot2, piece2)

        # --- 按鍵長按處理 ---
        keys = pg.key.get_pressed()

        # P1 & P2 手動控制
        if not game_over1:
            if keys[pg.K_a] and key_ticker1[pg.K_a] == 0:
                key_ticker1[pg.K_a] = 6
                Handler.moveLeft(shot1, piece1)
            if keys[pg.K_d] and key_ticker1[pg.K_d] == 0:
                key_ticker1[pg.K_d] = 6
                Handler.moveRight(shot1, piece1)
            if keys[pg.K_s] and key_ticker1[pg.K_s] == 0:
                key_ticker1[pg.K_s] = 6
                Handler.drop(shot1, piece1)
#        if not game_over2:
#            if keys[pg.K_LEFT] and key_ticker2[pg.K_LEFT] == 0:
#                key_ticker2[pg.K_LEFT] = 6
#                Handler.moveLeft(shot2, piece2)
#            if keys[pg.K_RIGHT] and key_ticker2[pg.K_RIGHT] == 0:
#                key_ticker2[pg.K_RIGHT] = 6
#                Handler.moveRight(shot2, piece2)
#            if keys[pg.K_DOWN] and key_ticker2[pg.K_DOWN] == 0:
#                key_ticker2[pg.K_DOWN] = 6
#                Handler.drop(shot2, piece2)

        for k in key_ticker1:
            if key_ticker1[k] > 0: key_ticker1[k] -= 1
#        for k in key_ticker2:
#            if key_ticker2[k] > 0: key_ticker2[k] -= 1


        # --- P1 邏輯：方塊固定與攻擊流程 ---
        if not game_over1:
            if piece1.is_fixed:
                (clears, all_clear) = Handler.eliminateFilledRows(shot1, piece1)
                is_power_move = (clears == 4)
                if clears > 0:
                    shot1.combo_count += 1
                else:
                    shot1.combo_count = 0
                atk1 = Handler.calculateAttack(clears, shot1.combo_count, shot1.is_b2b, all_clear)
                if is_power_move:
                    shot1.is_b2b = True
                elif clears > 0:
                    shot1.is_b2b = False
                if atk1 > 0:
                    cancel_amt = min(atk1, shot1.pending_garbage)
                    shot1.pending_garbage -= cancel_amt
                    atk1 -= cancel_amt
                if atk1 > 0:
                    shot2.pending_garbage += atk1
                piece1, next_piece1 = next_piece1, getRandomPiece()
                if Handler.isDefeat(shot1, piece1):
                    game_over1 = True
                    print("Player 1 Game Over!!")
                    print("P1 Score:", shot1.score, "| P1 Lines:", shot1.line_count)

        # --- P2 邏輯：方塊固定與攻擊流程 ---
        # --- P2 邏輯：AI 決策、放置與攻擊流程 ---
        if not game_over2:
            # P2 輪到它了 (piece2.is_fixed 為 True 代表上一個方塊剛放好)
            if piece2.is_fixed: 

                # --- P2 方塊固定後的結算 (跟 P1 一樣) ---
                (clears, all_clear) = Handler.eliminateFilledRows(shot2, piece2)
                is_power_move = (clears == 4)
                if clears > 0:
                    shot2.combo_count += 1
                else:
                    shot2.combo_count = 0

                atk2 = Handler.calculateAttack(clears, shot2.combo_count, shot2.is_b2b, all_clear)

                if is_power_move:
                    shot2.is_b2b = True
                elif clears > 0:
                    shot2.is_b2b = False

                if atk2 > 0:
                    cancel_amt = min(atk2, shot2.pending_garbage)
                    shot2.pending_garbage -= cancel_amt
                    atk2 -= cancel_amt
                if atk2 > 0:
                    shot1.pending_garbage += atk2

                # --- 取得新方塊 ---
                piece2, next_piece2 = next_piece2, getRandomPiece()

                # --- 檢查是否因「新方塊」而死亡 (Spawn Die) ---
                if Handler.isDefeat(shot2, piece2):
                    game_over2 = True
                    print("Player 2 Game Over!!")
                    print("P2 Score:", shot2.score, "| P2 Lines:", shot2.line_count)

                # --- (★★ AI 核心 ★★) ---
                # 如果還沒死，就讓 AI 決定下一步
                if not game_over2:
                    # 1. 呼叫 AI 找到最佳移動
                    #    我們傳入 "copy" 是為了確保 AI 不會不小心改到真實的遊戲狀態
                    best_move = ai_player.find_best_move(copy.deepcopy(shot2), copy.deepcopy(piece2))

                    if best_move:
                        # 2. 應用 AI 的決策
                        (best_x, best_rotation) = best_move
                        piece2.x = best_x
                        piece2.rotation = best_rotation

                        # 3. 立即瞬降 (AI 不需要慢慢等)
                        Handler.instantDrop(shot2, piece2)
                    else:
                        # 如果 AI 找不到任何合法的移動 (例如盤面已滿)
                        # 就隨便放一個然後讓它死
                        Handler.instantDrop(shot2, piece2)

        # --- 遊戲結束判斷 ---
        if game_over1 and game_over2:
            run = False
        
        # --- 繪製畫面 ---
        screen.fill(config.background_color)

        if not game_over1:
            draw_player_ui(screen, shot1, piece1, next_piece1, myfont,
                           config.P1_OFFSET_X, config.P1_SCORE_POS, 
                           config.P1_LINE_POS, config.P1_NEXT_PIECE_POS,
                           config.P1_GARBAGE_BAR_POS)
        
        if not game_over2:
            draw_player_ui(screen, shot2, piece2, next_piece2, myfont,
                           config.P2_OFFSET_X, config.P2_SCORE_POS,
                           config.P2_LINE_POS, config.P2_NEXT_PIECE_POS,
                           config.P2_GARBAGE_BAR_POS)

        pg.display.update()
        fpsClock.tick(config.fps)
    
    print("----- Final Result -----")
    print(f"P1 Score: {shot1.score} | P1 Lines: {shot1.line_count}")
    print(f"P2 Score: {shot2.score} | P2 Lines: {shot2.line_count}")
    
    if game_over1 and not game_over2:
        print("Player 2 Wins!")
    elif game_over2 and not game_over1:
        print("Player 1 Wins!")
    else:
        print("It's a Draw!")
        
    pg.quit()
"""

if __name__ == "__main__":
    main()