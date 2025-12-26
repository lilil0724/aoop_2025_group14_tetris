import pygame as pg

# Global settings
AI_MOVE_DELAY = 5 
AI_THINKING_DELAY = 10 
SHOW_GHOST = True  # 預設開啟投影

#  AI 速度等級 (1: Slow, 2: Normal, 3: Fast, 4: God/Instant)
AI_SPEED_LEVEL = 2

VOLUME = 0.5 # 預設音量

# Key Bindings
KEY_BINDINGS = {
    'P1_LEFT': pg.K_a,
    'P1_RIGHT': pg.K_d,
    'P1_DOWN': pg.K_s,
    'P1_ROTATE': pg.K_w,
    'P1_ROTATE_CCW': pg.K_q,
    'P1_DROP': pg.K_LSHIFT,
    
    'P2_LEFT': pg.K_LEFT,
    'P2_RIGHT': pg.K_RIGHT,
    'P2_DOWN': pg.K_DOWN,
    'P2_ROTATE': pg.K_UP,
    'P2_ROTATE_CCW': pg.K_l,
    'P2_DROP': pg.K_RSHIFT
}
