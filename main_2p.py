import pygame as pg
import pieces
import shots
import config
import Handler
import random
import ai_agent  # 新增

USE_AI_P2 = True  # 讓 P2 交給 AI
DEBUG = False

init_start = (5, 0)  # 放置新方塊的位置 (邏輯座標)


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


def draw_player_ui(screen, shot, piece, next_piece, font,
                   offset_x, score_pos, line_pos, next_piece_pos,
                   garbage_bar_pos):

    # 1. 更新 shot 狀態 (邏輯)
    for y in range(config.rows):
        for x in range(config.columns):
            if shot.status[y][x] != 2:
                shot.status[y][x] = 0

    for y, x in Handler.getCellsAbsolutePosition(piece):
        if 0 <= y < config.rows and 0 <= x < config.columns:
            shot.color[y][x] = piece.color
            shot.status[y][x] = 1

    # 2. 繪製遊戲板
    for y, line in enumerate(shot.color):
        for x, color in enumerate(line):
            draw_color = (0, 0, 0) if shot.status[y][x] == 0 else color
            pg.draw.rect(screen, draw_color, (
                offset_x + x * config.grid,
                y * config.grid,
                config.grid,
                config.grid
            ))

    # 3. 分數
    textsurface = font.render(f'Score: {shot.score}', False, (255, 255, 255))
    screen.blit(textsurface, score_pos)

    # 4. 行數
    textsurface = font.render(f'Line: {shot.line_count}', False, (255, 255, 255))
    screen.blit(textsurface, line_pos)

    # 5. 下一個方塊背景
    for y in range(-2, 3):
        for x in range(-2, 3):
            pg.draw.rect(screen, (50, 50, 50), (
                next_piece_pos[0] + x * config.grid,
                next_piece_pos[1] + y * config.grid,
                config.grid,
                config.grid
            ))

    # 6. 下一個方塊
    for y, x in next_piece.getCells():
        color = next_piece.color
        pg.draw.rect(screen, color, (
            next_piece_pos[0] + x * config.grid,
            next_piece_pos[1] + y * config.grid,
            config.grid,
            config.grid
        ))

    # 7. 垃圾條顯示
    if shot.pending_garbage > 0:
        bar_max_height = config.height * 0.9
        bar_y_start = config.height * 0.05
        pending_visual = min(shot.pending_garbage, 12)
        bar_height = bar_max_height * (pending_visual / 12.0)
        bar_x = garbage_bar_pos[0]
        bar_y_fill = (bar_y_start + bar_max_height) - bar_height
        pg.draw.rect(screen, (80, 80, 80), (bar_x, bar_y_start, config.GARBAGE_BAR_WIDTH, bar_max_height))
        pg.draw.rect(screen, (255, 50, 50), (bar_x, bar_y_fill, config.GARBAGE_BAR_WIDTH, bar_height))

    draw_grid(screen, offset_x)


def main():
    pg.init()
    pg.font.init()
    myfont = pg.font.SysFont(*config.font)
    fpsClock = pg.time.Clock()
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris 1v1")

    # --- 初始化 ---
    shot1, piece1, next_piece1 = shots.Shot(), getRandomPiece(), getRandomPiece()
    shot2, piece2, next_piece2 = shots.Shot(), getRandomPiece(), getRandomPiece()

    counter1 = counter2 = 0
    key_ticker1 = {pg.K_a: 0, pg.K_s: 0, pg.K_d: 0}
    key_ticker2 = {pg.K_LEFT: 0, pg.K_DOWN: 0, pg.K_RIGHT: 0}
    game_over1 = game_over2 = False

    ai_cmd_queue = []
    need_new_plan = True

    run = True
    while run:
        # --- 自動下落 ---
        if not DEBUG:
            if not game_over1:
                counter1 = (counter1 + 1) % (config.difficulty + 1)
                if counter1 == 0:
                    Handler.drop(shot1, piece1)
            if not game_over2:
                counter2 = (counter2 + 1) % (config.difficulty + 1)
                if counter2 == 0:
                    Handler.drop(shot2, piece2)

        # --- 垃圾行插入 ---
        for shot, piece, g_over, label in [
            (shot1, piece1, game_over1, "Player 1"),
            (shot2, piece2, game_over2, "Player 2")
        ]:
            if not g_over and shot.pending_garbage > 0:
                shot.garbage_insert_timer += 1
                if shot.garbage_insert_timer > config.GARBAGE_INSERT_DELAY:
                    lines_to_add = min(shot.pending_garbage, config.GARBAGE_LINES_PER_INSERT)
                    Handler.insertGarbage(shot, lines_to_add)
                    shot.pending_garbage -= lines_to_add
                    shot.garbage_insert_timer = 0
                    if Handler.isDefeat(shot, piece):
                        if label == "Player 1":
                            game_over1 = True
                        else:
                            game_over2 = True
                        print(f"{label} Game Over!! (Killed by garbage)")

        # --- 事件處理 ---
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    run = False
                if not game_over1:
                    if event.key == pg.K_w: Handler.rotate(shot1, piece1)
                    if event.key == pg.K_a: Handler.moveLeft(shot1, piece1)
                    if event.key == pg.K_d: Handler.moveRight(shot1, piece1)
                    if event.key == pg.K_s: Handler.drop(shot1, piece1)
                    if event.key == pg.K_LSHIFT: Handler.instantDrop(shot1, piece1)

        # --- AI 控制 P2 ---
        if USE_AI_P2 and not game_over2:
            if need_new_plan or piece2.is_fixed:
                target_rot, target_x, emergency = ai_agent.plan_for_piece(shot2, pieces.Piece(0, 0, piece2.shape))
                dummy_piece = pieces.Piece(piece2.x, piece2.y, piece2.shape)
                dummy_piece.rotation = piece2.rotation
                ai_cmd_queue = ai_agent.next_move_commands(dummy_piece, target_rot, target_x)
                need_new_plan = False

            step_per_frame = 2
            for _ in range(min(step_per_frame, len(ai_cmd_queue))):
                cmd = ai_cmd_queue.pop(0)
                if cmd == "ROT": Handler.rotate(shot2, piece2)
                elif cmd == "L": Handler.moveLeft(shot2, piece2)
                elif cmd == "R": Handler.moveRight(shot2, piece2)
                elif cmd == "DROP": Handler.instantDrop(shot2, piece2)

        # --- P1 固定處理 ---
        if not game_over1 and piece1.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot1, piece1)
            atk1 = Handler.calculateAttack(clears, shot1.combo_count, shot1.is_b2b, all_clear)
            if atk1 > 0:
                cancel_amt = min(atk1, shot1.pending_garbage)
                shot1.pending_garbage -= cancel_amt
                shot2.pending_garbage += atk1 - cancel_amt
            piece1, next_piece1 = next_piece1, getRandomPiece()
            if Handler.isDefeat(shot1, piece1):
                game_over1 = True

        # --- P2 固定處理 ---
        if not game_over2 and piece2.is_fixed:
            clears, all_clear = Handler.eliminateFilledRows(shot2, piece2)
            atk2 = Handler.calculateAttack(clears, shot2.combo_count, shot2.is_b2b, all_clear)
            if atk2 > 0:
                cancel_amt = min(atk2, shot2.pending_garbage)
                shot2.pending_garbage -= cancel_amt
                shot1.pending_garbage += atk2 - cancel_amt
            piece2, next_piece2 = next_piece2, getRandomPiece()
            if Handler.isDefeat(shot2, piece2):
                game_over2 = True
            need_new_plan = True

        # --- 遊戲結束判斷 ---
        if game_over1 and game_over2:
            run = False

        # --- 繪圖更新 ---
        screen.fill(config.background_color)
        draw_player_ui(screen, shot1, piece1, next_piece1, myfont,
                       config.P1_OFFSET_X, config.P1_SCORE_POS,
                       config.P1_LINE_POS, config.P1_NEXT_PIECE_POS,
                       config.P1_GARBAGE_BAR_POS)
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


if __name__ == "__main__":
    main()
