import config

class Piece(object):
    def __init__(self, x, y, shape): # <-- 修改 (原為 row, column)
        self.x = x  # center position (column)
        self.y = y  # center position (row)
        self.shape = shape
        self.rotation = 0
        self.color = config.shape_colors[shape]
        self.is_fixed = False

    # def getColor(self): <-- 移除
    #     return config.shape_colors[self.shape]

    def getCells(self):
        shapes = config.shapes[self.shape]
        return shapes[self.rotation % len(shapes)]