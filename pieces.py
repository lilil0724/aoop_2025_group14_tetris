import config


class Piece(object):
    def __init__(self, row, column, shape):
        self.x = row  # center position
        self.y = column  # center position
        self.shape = shape
        self.rotation = 0
        self.color = config.shape_colors[shape]
        self.is_fixed = False

    def getColor(self):
        return config.shape_colors[self.shape]

    def getCells(self):
        shapes = config.shapes[self.shape]
        return shapes[self.rotation % len(shapes)]
