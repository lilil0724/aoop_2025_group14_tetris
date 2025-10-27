import pygame as pg
import pieces
import shots
import config
import Handler
import random

DEBUG = False

init_start = (5, 0) # 放置新方塊的位置 (邏輯座標)


def getRandomPiece():
    shape = random.choice(list(config.shapes.keys()))
    piece = pieces.Piece(*init_start, shape)
    return piece

def draw_grid(surface, offset_x): # <-- 增加 offset_x 參數
    """半透明灰色格線"""
    grid_surface = pg.Surface(
        (config.columns * config.grid, config.rows * config.grid), pg.SRCALPHA
    )
    # 顏色最後一個值是 alpha (透明度)，0=全透明，255=不透明
    color = (150, 150, 150, 60)  # 灰白色 + 淡淡透明感
    for y in range(config.rows):
        pg.draw.line(grid_surface, color, (0, y * config.grid), (config.columns * config.grid, y * config.grid))
    for x in range(config.columns):
        pg.draw.line(grid_surface, color, (x * config.grid, 0), (x * config.grid, config.rows * config.grid))
    surface.blit(grid_surface, (offset_x, 0)) # <-- 使用 offset_x


# 'update' 函數被重構為 'draw_player_ui'
def draw_player_ui(screen, shot, piece, next_piece, font, offset_x, score_pos, line_pos, next_piece_pos):
    # screen.fill(config.background_color) # <-- 這將移到主迴圈中
    
    # 1. 更新 shot 狀態 (邏輯)
    for y in range(config.rows):
        for x in range(config.columns):
            if shot.status[y][x] != 2:
                shot.status[y][x] = 0

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
                offset_x + x * config.grid, # <-- 加上 offset_x
                y * config.grid,
                config.grid,
                config.grid
            ))

    # 3. 繪製分數
    textsurface = font.render('Score: {}'.format(
        shot.score), False, (255, 255, 255))
    screen.blit(textsurface, score_pos) # <-- 使用傳入的 pos

    # 4. 繪製行數
    textsurface = font.render('Line: {}'.format(
        shot.line_count), False, (255, 255, 255))
    screen.blit(textsurface, line_pos) # <-- 使用傳入的 pos

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
    
    # 7. 繪製網格
    draw_grid(screen, offset_x)


def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height)) # <-- 使用新的寬度
    pg.display.set_caption("Tetris 1v1")

    # --- P1 遊戲狀態 ---
    shot1 = shots.Shot()
    piece1 = getRandomPiece()
    next_piece1 = getRandomPiece()
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0} # 初始化 P1 按鍵
    game_over1 = False

    # --- P2 遊戲狀態 ---
    shot2 = shots.Shot()
    piece2 = getRandomPiece()
    next_piece2 = getRandomPiece()
    counter2 = 0
    key_ticker2 = {pg.K_LEFT: 0, pg.K_DOWN: 0, pg.K_RIGHT: 0} # 初始化 P2 按鍵
    game_over2 = False

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
            
            if not game_over2:
                if counter2 == config.difficulty:
                    Handler.drop(shot2, piece2)
                    counter2 = 0
                else:
                    counter2 += 1

        # --- 事件處理 (單次按鍵) ---
        for event in pg.event.get(): # 檢查各種事件
            if event.type == pg.QUIT: # 關閉視窗
                run = False
            elif event.type == pg.KEYDOWN: # 遍歷所有按下鍵盤的事件並觸發動作
                if event.key == pg.K_ESCAPE: # 按下 esc
                    run = False
                
                # --- P1 (WASD) ---
                if not game_over1:
                    if event.key == pg.K_w: # P1 旋轉
                        Handler.rotate(shot1, piece1)
                    if event.key == pg.K_s: # P1 下降
                        key_ticker1[pg.K_s] = 13
                        Handler.drop(shot1, piece1)
                    if event.key == pg.K_a: # P1 向左
                        key_ticker1[pg.K_a] = 13
                        Handler.moveLeft(shot1, piece1)
                    if event.key == pg.K_d: # P1 向右
                        key_ticker1[pg.K_d] = 13
                        Handler.moveRight(shot1, piece1)
                    if event.key == pg.K_LSHIFT: # P1 瞬間下降 (使用 Left Shift)
                        Handler.instantDrop(shot1, piece1)

                # --- P2 (方向鍵) ---
                if not game_over2:
                    if event.key == pg.K_UP: # P2 旋轉
                        Handler.rotate(shot2, piece2)
                    if event.key == pg.K_DOWN: # P2 下降
                        key_ticker2[pg.K_DOWN] = 13
                        Handler.drop(shot2, piece2)
                    if event.key == pg.K_LEFT: # P2 向左
                        key_ticker2[pg.K_LEFT] = 13
                        Handler.moveLeft(shot2, piece2)
                    if event.key == pg.K_RIGHT: # P2 向右
                        key_ticker2[pg.K_RIGHT] = 13
                        Handler.moveRight(shot2, piece2)
                    if event.key == pg.K_SPACE: # P2 瞬間下降
                        Handler.instantDrop(shot2, piece2)

        # --- 按鍵長按處理 ---
        keys = pg.key.get_pressed()
        
        # --- P1 長按 ---
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

        # --- P2 長按 ---
        if not game_over2:
            if keys[pg.K_LEFT] and key_ticker2[pg.K_LEFT] == 0:
                key_ticker2[pg.K_LEFT] = 6
                Handler.moveLeft(shot2, piece2)
            if keys[pg.K_RIGHT] and key_ticker2[pg.K_RIGHT] == 0:
                key_ticker2[pg.K_RIGHT] = 6
                Handler.moveRight(shot2, piece2)
            if keys[pg.K_DOWN] and key_ticker2[pg.K_DOWN] == 0:
                key_ticker2[pg.K_DOWN] = 6
                Handler.drop(shot2, piece2)

        # --- 更新 Ticker ---
        for k in key_ticker1:
            if key_ticker1[k] > 0:
                key_ticker1[k] -= 1
        for k in key_ticker2:
            if key_ticker2[k] > 0:
                key_ticker2[k] -= 1

        # --- P1 邏輯：方塊固定與失敗判斷 ---
        if not game_over1:
            if piece1.is_fixed: # 正把方塊固定時觸發的動作
                Handler.eliminateFilledRows(shot1, piece1)
                piece1, next_piece1 = next_piece1, getRandomPiece()
            
            if Handler.isDefeat(shot1, piece1):
                game_over1 = True
                print("Player 1 Game Over!!")
                print("P1 Score:", shot1.score)
                print("P1 Eliminated line:", shot1.line_count)

        # --- P2 邏輯：方塊固定與失敗判斷 ---
        if not game_over2:
            if piece2.is_fixed: # 正把方塊固定時觸發的動作
                Handler.eliminateFilledRows(shot2, piece2)
                piece2, next_piece2 = next_piece2, getRandomPiece()
            
            if Handler.isDefeat(shot2, piece2):
                game_over2 = True
                print("Player 2 Game Over!!")
                print("P2 Score:", shot2.score)
                print("P2 Eliminated line:", shot2.line_count)

        # --- 遊戲結束判斷 ---
        if game_over1 and game_over2:
            run = False

        # --- 繪製畫面 ---
        screen.fill(config.background_color) # 統一清除背景

        if not game_over1:
            draw_player_ui(screen, shot1, piece1, next_piece1, myfont,
                           config.P1_OFFSET_X, config.P1_SCORE_POS, 
                           config.P1_LINE_POS, config.P1_NEXT_PIECE_POS)
        
        if not game_over2:
            draw_player_ui(screen, shot2, piece2, next_piece2, myfont,
                           config.P2_OFFSET_X, config.P2_SCORE_POS,
                           config.P2_LINE_POS, config.P2_NEXT_PIECE_POS)

        pg.display.update()
        fpsClock.tick(config.fps)
    
    print("----- Final Result -----")
    print(f"P1 Score: {shot1.score} | P1 Lines: {shot1.line_count}")
    print(f"P2 Score: {shot2.score} | P2 Lines: {shot2.line_count}")
    pg.quit()


if __name__ == "__main__":
    main()