import pygame as pg
import pieces
import shots
import config
import Handler
import random

DEBUG = False
init_start = (5, 10) if DEBUG else (5, 0)

def getRandomPiece():
    shape = random.choice(list(config.shapes.keys()))
    piece = pieces.Piece(*init_start, shape)
    return piece

def draw_grid(surface):
    """半透明灰色格線"""
    grid_surface = pg.Surface(
        (config.columns * config.grid, config.rows * config.grid), pg.SRCALPHA
    )
    color = (150, 150, 150, 60)
    for y in range(config.rows):
        pg.draw.line(grid_surface, color, (0, y * config.grid), (config.columns * config.grid, y * config.grid))
    for x in range(config.columns):
        pg.draw.line(grid_surface, color, (x * config.grid, 0), (x * config.grid, config.rows * config.grid))
    surface.blit(grid_surface, (0, 0))

# --- 修改：增加 level 和 speed 參數 ---
def update(screen, shot, piece, next_piece, font, level, speed):
    screen.fill(config.background_color)

    # 繪製遊戲區內的方塊 (已固定的和當前的)
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
            pg.draw.rect(screen, color, (x * config.grid, y * config.grid, config.grid, config.grid))

    # --- 修正與新增：確保所有文字都被正確繪製 ---
    # 分數
    textsurface = font.render(f'Score: {shot.score}', False, (255, 255, 255))
    screen.blit(textsurface, config.score_pos)

    # 行數
    textsurface = font.render(f'Line: {shot.line_count}', False, (255, 255, 255))
    screen.blit(textsurface, config.line_pos)

    # 等級
    textsurface = font.render(f'Level: {level}', False, (255, 255, 255))
    screen.blit(textsurface, config.level_pos)
    
    # 速度
    textsurface = font.render(f'Speed: {speed}', False, (255, 255, 255))
    screen.blit(textsurface, config.speed_pos)
    # -----------------------------------------------

    # 下一個方塊的背景
    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (
                config.next_piece_pos[0] + x * config.grid,
                config.next_piece_pos[1] + y * config.grid,
                config.grid,
                config.grid
            ))

    # 下一個方塊
    for y, x in next_piece.getCells():
        pg.draw.rect(screen, next_piece.color, (
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

    run = True
    counter = 0
    key_ticker = {}
    
    level = 1
    current_difficulty = config.difficulty

    # --- 修改：第一次呼叫 update 時，傳入所有必要的參數 ---
    update(screen, shot, piece, next_piece, myfont, level, current_difficulty)

    while run:
        if not DEBUG and counter >= current_difficulty:
            Handler.drop(shot, piece)
            counter = 0

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    run = False
                if event.key == pg.K_UP:
                    Handler.rotate(shot, piece)
                if event.key == pg.K_DOWN:
                    key_ticker[pg.K_DOWN] = 13
                    Handler.drop(shot, piece)
                    counter = 0
                if event.key == pg.K_LEFT:
                    key_ticker[pg.K_LEFT] = 13
                    Handler.moveLeft(shot, piece)
                if event.key == pg.K_RIGHT:
                    key_ticker[pg.K_RIGHT] = 13
                    Handler.moveRight(shot, piece)
                if event.key == pg.K_SPACE:
                    Handler.instantDrop(shot, piece)
                    counter = 0

        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] and key_ticker.get(pg.K_LEFT) == 0:
            key_ticker[pg.K_LEFT] = 6
            Handler.moveLeft(shot, piece)
        if keys[pg.K_RIGHT] and key_ticker.get(pg.K_RIGHT) == 0:
            key_ticker[pg.K_RIGHT] = 6
            Handler.moveRight(shot, piece)
        if keys[pg.K_DOWN] and key_ticker.get(pg.K_DOWN) == 0:
            key_ticker[pg.K_DOWN] = 6
            Handler.drop(shot, piece)
            counter = 0
            
        for k in list(key_ticker.keys()):
            if key_ticker[k] > 0:
                key_ticker[k] -= 1

        if piece.is_fixed:
            lines_before = shot.line_count
            Handler.eliminateFilledRows(shot, piece)
            lines_after = shot.line_count

            if lines_after > lines_before:
                new_level = (lines_after // config.lines_per_level) + 1
                if new_level > level:
                    level = new_level
                    print(f"Level Up! Current Level: {level}")
                    new_difficulty = config.difficulty - (level - 1) * config.speed_increment
                    current_difficulty = max(new_difficulty, config.min_difficulty)
                    print(f"New speed (frames per drop): {current_difficulty}")

            piece, next_piece = next_piece, getRandomPiece()
            
        if not Handler.isDefeat(shot, piece):
            # --- 修改：在主迴圈末端呼叫 update，並傳入所有參數 ---
            update(screen, shot, piece, next_piece, myfont, level, current_difficulty)
        else:
            run = False
            
        pg.display.update()
        fpsClock.tick(config.fps)
        counter += 1
        
    print("Game Over!!")
    print(f"Score: {shot.score}")
    print(f"Eliminated line: {shot.line_count}")
    print(f"Final Level: {level}")
    pg.quit()

if __name__ == "__main__":
    main()
