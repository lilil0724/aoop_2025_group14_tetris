import config

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
        