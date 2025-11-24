
import pygame as pg
import pieces
import shots
import config
import Handler
import random
import copy

from ai_player_nn import AIPlayerNN

DEBUG = False

init_start = (5, 0) # 放置新方塊的位置 (邏輯座標)
USE_TRAINED_AI_NN = True

def getRandomPiece():
    shape = random.choice(list(config.shapes.keys()))
    piece = pieces.Piece(*init_start, shape)
    return piece

def draw_grid(surface, offset_x):
    grid_surface = pg.Surface(
        (config.columns * config.grid, config.rows * config.grid), pg.SRCALPHA
    )
    color = (150, 150, 150, 60)
    for y in range(config.rows):
        pg.draw.line(grid_surface, color, (0, y * config.grid), (config.columns * config.grid, y * config.grid))
    for x in range(config.columns):
        pg.draw.line(grid_surface, color, (x * config.grid, 0), (x * config.grid, config.rows * config.grid))
    surface.blit(grid_surface, (offset_x, 0))

def draw_player_ui(screen, shot, piece, next_piece, font, 
                   offset_x, score_pos, line_pos, next_piece_pos, 
                   garbage_bar_pos): 
    for y in range(config.rows):
        for x in range(config.columns):
            if shot.status[y][x] != 2:
                shot.status[y][x] = 0

    if not piece.is_fixed:
        for y, x in Handler.getCellsAbsolutePosition(piece):
            if 0 <= y < config.rows and 0 <= x < config.columns:
                shot.color[y][x] = piece.color
                shot.status[y][x] = 1

    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            draw_color = (0, 0, 0) if shot.status[y][x] == 0 else color
            pg.draw.rect(screen, draw_color, (
                offset_x + x * config.grid,
                y * config.grid,
                config.grid,
                config.grid
            ))

    textsurface = font.render('Score: {}'.format(shot.score), False, (255, 255, 255))
    screen.blit(textsurface, score_pos)

    textsurface = font.render('Line: {}'.format(shot.line_count), False, (255, 255, 255))
    screen.blit(textsurface, line_pos)

    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (
                next_piece_pos[0] + x * config.grid,
                next_piece_pos[1] + y * config.grid,
                config.grid,
                config.grid
            ))

    for y, x in next_piece.getCells():
        color = next_piece.color
        pg.draw.rect(screen, color, (
            next_piece_pos[0] + x * config.grid,
            next_piece_pos[1] + y * config.grid,
            config.grid,
            config.grid
        ))

    if shot.pending_garbage > 0:
        bar_max_height = config.height * 0.9
        bar_y_start = config.height * 0.05
        pending_visual = min(shot.pending_garbage, 12) 
        bar_fill_ratio = pending_visual / 12.0
        bar_height = bar_max_height * bar_fill_ratio
        bar_x = garbage_bar_pos[0]
        bar_y_fill = (bar_y_start + bar_max_height) - bar_height
        pg.draw.rect(screen, (80, 80, 80), (
            bar_x, bar_y_start, config.GARBAGE_BAR_WIDTH, bar_max_height
        ))
        pg.draw.rect(screen, (255, 50, 50), (
            bar_x, bar_y_fill, config.GARBAGE_BAR_WIDTH, bar_height
        ))

    draw_grid(screen, offset_x)

def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris 1v1 (NN Agent)")

    # AI 載入
    if USE_TRAINED_AI_NN:
        ai_nn = AIPlayerNN.load('tetris_valuenet.pt')
    else:
        ai_nn = AIPlayerNN()

    # P1 (人)
    shot1 = shots.Shot()
    piece1 = getRandomPiece()
    next_piece1 = getRandomPiece()
    counter1 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    game_over1 = False

    # P2 (AI)
    shot2 = shots.Shot()
    piece2 = getRandomPiece()
    next_piece2 = getRandomPiece()
    game_over2 = False

    # AI 先把第一顆放下
    if not game_over2:
        # 修改：加入 next_piece2 參數
        best_move = ai_nn.find_best_move(
            copy.deepcopy(shot2), 
            copy.deepcopy(piece2),
            copy.deepcopy(next_piece2) # <--- 新增這行
        )
        if best_move:
            piece2.x, piece2.rotation = best_move
            Handler.instantDrop(shot2, piece2)
        else:
            Handler.instantDrop(shot2, piece2)

    run = True
    while run:
        if not game_over1:
            if counter1 == config.difficulty:
                Handler.drop(shot1, piece1)
                counter1 = 0
            else:
                counter1 += 1

        # 垃圾行插入 (與你的 main_2p.py 相同邏輯)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    run = False
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
                    if event.key == pg.K_LSHIFT:
                        Handler.instantDrop(shot1, piece1)

        keys = pg.key.get_pressed()
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

        # P1 結算
        if not game_over1 and piece1.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot1, piece1)
            is_power_move = (clears == 4)
            if clears > 0: shot1.combo_count += 1
            else: shot1.combo_count = 0
            atk1 = Handler.calculateAttack(clears, shot1.combo_count, shot1.is_b2b, all_clear)
            if is_power_move: shot1.is_b2b = True
            elif clears > 0: shot1.is_b2b = False
            if atk1 > 0:
                cancel_amt = min(atk1, shot1.pending_garbage)
                shot1.pending_garbage -= cancel_amt
                atk1 -= cancel_amt
            if atk1 > 0:
                shot2.pending_garbage += atk1
            piece1, next_piece1 = next_piece1, getRandomPiece()
            if Handler.isDefeat(shot1, piece1):
                game_over1 = True

        # P2 用 NN 下子
        if not game_over2 and piece2.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot2, piece2)
            is_power_move = (clears == 4)
            if clears > 0: shot2.combo_count += 1
            else: shot2.combo_count = 0
            atk2 = Handler.calculateAttack(clears, shot2.combo_count, shot2.is_b2b, all_clear)
            if is_power_move: shot2.is_b2b = True
            elif clears > 0: shot2.is_b2b = False
            if atk2 > 0:
                cancel_amt = min(atk2, shot2.pending_garbage)
                shot2.pending_garbage -= cancel_amt
                atk2 -= cancel_amt
            if atk2 > 0:
                shot1.pending_garbage += atk2

            piece2, next_piece2 = next_piece2, getRandomPiece()
            if Handler.isDefeat(shot2, piece2):
                game_over2 = True

            if not game_over2:
                # 修改：加入 next_piece2 參數
                best_move = ai_nn.find_best_move(
                    copy.deepcopy(shot2), 
                    copy.deepcopy(piece2),
                    copy.deepcopy(next_piece2) # <--- 新增這行
                )
                if best_move:
                    piece2.x, piece2.rotation = best_move
                    Handler.instantDrop(shot2, piece2)
                else:
                    Handler.instantDrop(shot2, piece2)

        # 繪製
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

    pg.quit()

if __name__ == "__main__":
    main()
