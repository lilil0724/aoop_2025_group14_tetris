import config


def getCellsAbsolutePosition(piece):
    '''取得方塊當前所有方格的座標'''
    return [(y + piece.y, x + piece.x) for y, x in piece.getCells()]

# 印出目前方塊的所有方格的座標
def printPiece(shot, piece):
    print('目前方塊:', getCellsAbsolutePosition(piece))

### Let's practice NOW!

def moveLeft(shot, piece):
    move=True
    for y,x in getCellsAbsolutePosition(piece):
        if x==0:
            move=False
    if move:
        piece.x-=1
    pass

def moveRight(shot, piece):
    move=True
    for y,x in getCellsAbsolutePosition(piece):
        if x==9:
            move=False
    if move:
        piece.x+=1
    pass

def moveUp(shot, piece):
    move=True
    for y,x in getCellsAbsolutePosition(piece):
        if  y==1:
            move=False
    if move:
        piece.y-=1
    pass

def moveDown(shot, piece):
    move=True
    for y,x in getCellsAbsolutePosition(piece):
        if y==17:
            move=False
    if move:
        piece.y+=1
    pass

def printMap(shot, piece):
    print('Print Map:')
    print('---'*4)
    for y in range(20):
        for x in range(10):
            print(shot.status[y][x],end='')
        print()
    print('---'*4)
