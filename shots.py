import config
import random # <-- 新增

class Shot(object):
    def __init__(self):
        self.color = [[ config.background_color for _ in range(config.columns)] for __ in range(config.rows) ] # 2D-array representing the color of each cell
        
        '''
        2D-array representing the status of each cell
        0: empty
        1: the moving piece
        2: fixed pieces
        '''
        self.status = [[ 0 for _ in range(config.columns)] for __ in range(config.rows) ] 
        self.line_count = 0
        self.score = 0
        # self.speed = 1 (移除)
        
        # --- (新增) 1v1 攻擊/垃圾行狀態 ---
        self.pending_garbage = 0       # 待處理的垃圾行總數 (整數)
        self.combo_count = 0           # 目前的連擊 (REN) 次數
        self.is_b2b = False            # *上一手* 是否為威力技 (Tetris)
        self.garbage_insert_timer = 0  # 垃圾行插入的計時器
        # 垃圾洞位產生器 (RNG) 的狀態
        self.garbage_hole_pos = random.randint(0, config.columns - 1) 
        # --- (新增結束) ---