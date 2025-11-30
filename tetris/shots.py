import config
import random

class Shot(object):
    def __init__(self):
        self.color = [[ config.background_color for _ in range(config.columns)] for __ in range(config.rows) ] 
        self.status = [[ 0 for _ in range(config.columns)] for __ in range(config.rows) ] 
        self.line_count = 0
        self.score = 0
        self.speed = 1
        
        # 1v1 攻擊/垃圾行狀態
        self.pending_garbage = 0       
        self.combo_count = 0           
        self.is_b2b = False            
        self.garbage_insert_timer = 0  
        self.garbage_hole_pos = random.randint(0, config.columns - 1) 
        self.shake_timer = 0
        
        self.all_clear_timer = 0