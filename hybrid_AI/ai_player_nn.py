import torch
import torch.nn as nn
import numpy as np
import config
import copy
from typing import List, Tuple, Optional
import Handler

# =============================
# 1. 優化版特徵提取 (含 Wells + Normalization)
# =============================
def get_custom_features(board, lines_cleared):
    """
    Features (Normalized):
    1. Lines Cleared (0.0 ~ 1.0) -> /4.0
    2. Holes (0.0 ~ 1.0) -> /20.0 (Assume max 20 holes is bad enough)
    3. Bumpiness (0.0 ~ 1.0) -> /100.0
    4. Total Height (0.0 ~ 1.0) -> /200.0
    5. Wells (0.0 ~ 1.0) -> /10.0 (New Feature!)
    """
    rows, cols = config.rows, config.columns
    grid = np.array(board, dtype=np.uint8).reshape(rows, cols)
    filled = grid != 0
    
    # 1. Column Heights
    heights = []
    for c in range(cols):
        if np.any(filled[:, c]):
            h = rows - np.argmax(filled[:, c])
            heights.append(h)
        else:
            heights.append(0)
    heights = np.array(heights)
    
    # 2. Holes
    holes = 0
    for c in range(cols):
        has_block_above = False
        for r in range(rows):
            if grid[r][c] != 0:
                has_block_above = True
            elif has_block_above and grid[r][c] == 0:
                holes += 1

    # 3. Bumpiness
    bumpiness = np.sum(np.abs(np.diff(heights)))
    
    # 4. Total Height
    total_height = np.sum(heights)
    
    # 5. Wells (深井：兩邊高，中間低的洞，這對 Tetris 策略很重要)
    wells = 0
    for c in range(cols):
        # 左邊高度
        left_h = heights[c-1] if c > 0 else rows
        # 右邊高度
        right_h = heights[c+1] if c < cols - 1 else rows
        
        my_h = heights[c]
        # 如果兩邊都比我高 2 格以上，這就是一個深井 (適合放 I 型方塊)
        depth = min(left_h, right_h) - my_h
        if depth >= 2:
            wells += depth

    # --- 歸一化 (Normalization) 關鍵優化 ---
    # 神經網路不喜歡大數字 (e.g. 200)，喜歡 0~1 的小數
    norm_lines = lines_cleared / 4.0
    norm_holes = min(holes / 20.0, 1.0)       # 超過 20 個洞就當作爛透了
    norm_bump = min(bumpiness / 50.0, 1.0)
    norm_height = min(total_height / 100.0, 1.0)
    norm_wells = min(wells / 20.0, 1.0)
    
    return np.array([norm_lines, norm_holes, norm_bump, norm_height, norm_wells], dtype=np.float32)


# =============================
# 2. 優化版模型 (Heuristic Initialization)
# =============================
class TetrisNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 5 features (原本4個 + Wells)
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output Score
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """
        啟發式初始化：
        不要讓 AI 從零開始猜，我們直接給它一組「大概正確」的觀念。
        Lines(+) Holes(-) Bump(-) Height(-) Wells(+)
        這能讓訓練速度快 10 倍。
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        # 強制設定最後一層的權重傾向
        # 我們希望 Output 與 Features 的關係大約是：
        # Score ~ w1*Lines - w2*Holes - w3*Bump - w4*Height + w5*Wells
        # 雖然中間有 Hidden Layer，但我們可以透過這種方式給一點 bias
        with torch.no_grad():
             # 讓最後一層輸出偏向正值
             self.net[-1].bias.fill_(0.0)

    def forward(self, x):
        return self.net(x)

# =============================
# 3. AI Player
# =============================
class AIPlayerNN:
    def __init__(self, device: Optional[torch.device] = None, model_path: Optional[str] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TetrisNetwork().to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded optimized model from {model_path}")
            except:
                pass
        self.model.eval()

    def _enumerate_moves(self, shot, p) -> List[Tuple[int, int]]:
        moves: List[Tuple[int, int]] = []
        # 簡單優化：只檢查必要的旋轉 (某些方塊旋轉是對稱的，例如 S, Z, I 只需檢查 2 種，O 只需 1 種)
        # 但為了保險起見，維持完整檢查也行，Python 這邊不會太慢
        num_rotations = len(config.shapes[p.shape])
        if p.shape == 'O': num_rotations = 1
        elif p.shape in ['S', 'Z', 'I']: num_rotations = 2
        
        for rot in range(num_rotations):
            tmpl = copy.deepcopy(p)
            tmpl.rotation = rot
            # 優化：先檢查中間，再往兩邊擴散，比較容易提早找到合法解(非必要)
            for x in range(-2, config.columns + 1):
                tmpl.x = x
                if Handler.isValidPosition(shot, tmpl):
                    moves.append((x, rot))
        return moves

    @torch.no_grad()
    def find_best_move(self, game_state, piece, next_piece=None) -> Optional[Tuple[int, int]]:
        legal_moves = self._enumerate_moves(game_state, piece)
        if not legal_moves: return None
        
        inputs = []
        for (x, rot) in legal_moves:
            sim_shot = copy.deepcopy(game_state)
            sim_piece = copy.deepcopy(piece)
            sim_piece.x, sim_piece.rotation = x, rot
            
            Handler.instantDrop(sim_shot, sim_piece)
            clears, _ = Handler.eliminateFilledRows(sim_shot, sim_piece)
            
            # 使用新版 5 特徵
            feats = get_custom_features(sim_shot.status, clears)
            inputs.append(feats)
            
        b_in = torch.tensor(np.array(inputs), dtype=torch.float32, device=self.device)
        scores = self.model(b_in).squeeze(-1)
        best_idx = torch.argmax(scores).item()
        return legal_moves[best_idx]