import pygame as pg
import pieces
import shots
import config_solo as config
import Handler
import random

DEBUG = False

init_start = (5, 10) if DEBUG else (5, 0) # 放置新方塊的位置


def getRandomPiece():
    shape = random.choice(list(config.shapes.keys()))
    piece = pieces.Piece(*init_start, shape)
    return piece

def draw_grid(surface):
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
    surface.blit(grid_surface, (0, 0))


def update(screen, shot, piece, next_piece, font):
    screen.fill(config.background_color)
    

    for y in range(config.rows):
        for x in range(config.columns):
            if shot.status[y][x] != 2:
                shot.status[y][x] = 0

    for y, x in Handler.getCellsAbsolutePosition(piece):
        if 0 <= y < config.rows and 0 <= x < config.columns:
            shot.color[y][x] = piece.color
            shot.status[y][x] = 1

    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            if shot.status[y][x] == 0:
                color = (0, 0, 0)
            pg.draw.rect(screen, color, (
                x * config.grid,
                y * config.grid,
                config.grid,
                config.grid
            ))

    # score
    textsurface = font.render('Score: {}'.format(
        shot.score), False, (255, 255, 255))
    screen.blit(textsurface, config.score_pos)

    # line
    textsurface = font.render('Line: {}'.format(
        shot.line_count), False, (255, 255, 255))
    screen.blit(textsurface, config.line_pos)

    textsurface = font.render('Speed: {}'.format(
        shot.speed), False, (255, 255, 255))
    screen.blit(textsurface, config.speed_pos)

    # next piece (background)
    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (
                config.next_piece_pos[0] + x * config.grid,
                config.next_piece_pos[1] + y * config.grid,
                config.grid,
                config.grid
            ))

    # next piece
    for y, x in next_piece.getCells():
        color = next_piece.color
        pg.draw.rect(screen, color, (
            config.next_piece_pos[0] + x * config.grid,
            config.next_piece_pos[1] + y * config.grid,
            config.grid,
            config.grid
        ))
    draw_grid(screen)


def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris")
    shot = shots.Shot()

    piece = getRandomPiece()
    next_piece = getRandomPiece()

    update(screen, shot, piece, next_piece, myfont)
    run = True
    counter = 0
    key_ticker = {}
    while run:
        if not DEBUG and counter == config.difficulty:
            Handler.drop(shot, piece)
            counter -= config.difficulty # 重設計時器
        for event in pg.event.get(): # 檢查各種事件
            if event.type == pg.QUIT: # 關閉視窗
                run = False
            elif event.type == pg.KEYDOWN: # 遍歷所有按下鍵盤的事件並觸發動作
                if event.key == pg.K_ESCAPE: # 按下 esc
                    run = False
                if event.key == pg.K_UP: # 按下上方向鍵時
                    Handler.rotate(shot, piece)
                if event.key == pg.K_DOWN: # 按下下方向鍵時
                    key_ticker[pg.K_DOWN] = 13
                    Handler.drop(shot, piece)
                if event.key == pg.K_LEFT: # 按下左方向鍵時
                    key_ticker[pg.K_LEFT] = 13
                    Handler.moveLeft(shot, piece)
                if event.key == pg.K_RIGHT: # 按下右方向鍵時
                    key_ticker[pg.K_RIGHT] = 13
                    Handler.moveRight(shot, piece)
                if event.key == pg.K_SPACE: # 按下空白鍵時
                    Handler.instantDrop(shot, piece)
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] and key_ticker[pg.K_LEFT] == 0: # 持續按著左方向鍵時
            key_ticker[pg.K_LEFT] = 6
            Handler.moveLeft(shot, piece)
        if keys[pg.K_RIGHT] and key_ticker[pg.K_RIGHT] == 0: # 持續按著右方向鍵時
            key_ticker[pg.K_RIGHT] = 6
            Handler.moveRight(shot, piece)
        if keys[pg.K_DOWN] and key_ticker[pg.K_DOWN] == 0: # 持續按著下方向鍵時
            key_ticker[pg.K_DOWN] = 6
            Handler.drop(shot, piece)
        for k in key_ticker.keys():
            if key_ticker[k] > 0:
                key_ticker[k] -= 1
        if piece.is_fixed: # 正把方塊固定時觸發的動作
            Handler.eliminateFilledRows(shot, piece)
            piece, next_piece = next_piece, getRandomPiece()
        if shot.line_count % config.line_to_speedup == 0 and shot.line_count != 0:
            shot.speed += 1
            config.difficulty -= config.speed_increment
            shot.line_count += 1 # 避免重複加速
        if not Handler.isDefeat(shot, piece):
            update(screen, shot, piece, next_piece, myfont)
        else:
            run = False
        pg.display.update()
        fpsClock.tick(config.fps)
        counter += 1
    print("Game Over!!")
    print("Score:", shot.score)
    print("Eliminated line:", shot.line_count)
    pg.quit()


if __name__ == "__main__":
    main()