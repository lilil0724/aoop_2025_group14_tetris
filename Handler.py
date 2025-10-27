import config


def getCellsAbsolutePosition(piece):
    '''取得方塊當前所有方格的座標'''
    return [(y + piece.y, x + piece.x) for y, x in piece.getCells()]


def fixPiece(shot, piece):
    '''固定已落地的方塊，並且在main中自動切到下一個方塊'''
    piece.is_fixed = True
    for y, x in getCellsAbsolutePosition(piece):
        shot.status[y][x] = 2
        shot.color[y][x] = piece.color


### Your homework below. Enjoy :) ###

# 向左移動
def moveLeft(shot, piece):
    move=True
    for y,x in getCellsAbsolutePosition(piece):
        if x==0 or shot.status[y][x-1]==2:
            move=False
    if move:
        piece.x-=1
    pass

# 向右移動
def moveRight(shot, piece):
    move=True
    for y,x in getCellsAbsolutePosition(piece):
        if x==9 or shot.status[y][x+1]==2:
            move=False
    if move:
        piece.x+=1
    pass

# 使方塊下落一格
def drop(shot, piece):
    move=True
    for y,x in getCellsAbsolutePosition(piece):
        if y>=0:
            if y==19 or shot.status[y+1][x]==2:
                move=False
                fixPiece(shot, piece)
    if move:
        piece.y+=1
    pass

# 瞬間掉落
def instantDrop(shot, piece):
    for i in range(20):
        for y,x in getCellsAbsolutePosition(piece):
            if y==19 or shot.status[y+1][x]==2:
                return
        piece.y+=1
    fixPiece(shot,piece)
    pass

# 旋轉方塊
def rotate(shot, piece):
    piece.rotation+=1
    for y,x in getCellsAbsolutePosition(piece):   
        if shot.status[y][x]==2 or x<0 or x>=10:
            piece.rotation-=1
            return
    pass

# 判斷是否死掉（出局）
def isDefeat(shot, piece):
    for y,x in getCellsAbsolutePosition(piece):
        if y>=0 and y==0 and shot.status[y+1][x]==2:
            return True
        else:
            return False
    pass

# 消去列
def eliminateFilledRows(shot, piece):
    m=0
    line=0
    score_count={}
    score_count[1]=40
    score_count[2]=100
    score_count[3]=300
    score_count[4]=1200
    for y in range(20):
        for i in range(10):
            if shot.status[y][i]==2:
                eliminate=True
                m=y
            else:
                eliminate=False
                break
        if eliminate:
            line+=1
            for x in range(10):
                shot.status[m][x]=0
            for y in range(m-1,0,-1):
                    for x in range(10):
                        shot.status[y+1][x]=shot.status[y][x]
    shot.line_count+=line
    shot.score+=score_count.get(line,0)
    pass

