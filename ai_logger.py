import json
import numpy as np
from datetime import datetime

DATA_FILE = "dataset.jsonl"

def record_sample(shot, piece, next_piece, target_rot, target_x, clears, atk):
    """
    儲存一筆遊戲資料
    """
    sample = {
        "board": np.array(shot.status).tolist(),  # 盤面 (20x10)
        "piece": piece.shape,                     # 當前方塊種類
        "next_piece": next_piece.shape,           # 下一個方塊種類
        "target_rot": target_rot,                 # 目標旋轉
        "target_x": target_x,                     # 目標 X 位置
        "clears": clears,                         # 清除的行數
        "attack": atk,                            # 攻擊行數
        "timestamp": datetime.now().isoformat()
    }
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")
