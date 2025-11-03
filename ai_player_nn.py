import copy
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# 專案內部模組（沿用你原本的專案結構）
import config
import pieces
import Handler


# =============================
# 盤面特徵擷取（NumPy 加速版）
# =============================
def _extract_features_from_board_np(board) -> List[float]:
    """
    高速特徵擷取：回傳內容與舊版 _extract_features_from_board 相同順序/長度。
    [flattened grid] + [column heights(10)] + [adjacent deltas(9)] +
    [holes, aggregate_height, bumpiness, max_height, complete_lines, wells]
    """
    rows, cols = config.rows, config.columns
    grid = np.array(board, dtype=np.uint8).reshape(rows, cols)
    flat = grid.reshape(-1).astype(np.float32).tolist()

    filled = grid != 0
    any_filled = filled.any(axis=0)  # shape (cols,)
    first_filled = np.argmax(filled, axis=0)  # index of first True per col (0 if none True)
    heights = np.where(any_filled, rows - first_filled, 0).astype(np.float32)  # 高度向量，長度=cols

    deltas = np.abs(np.diff(heights)).astype(np.float32)  # 相鄰列高度差，長度=cols-1

    # 洞：上方有方塊的空格
    above = np.maximum.accumulate(filled, axis=0)
    holes = float(np.sum((~filled) & above))

    aggregate_height = float(np.sum(heights))
    bumpiness = float(np.sum(deltas))
    max_height = float(np.max(heights)) if heights.size else 0.0
    complete_lines = float(np.sum(np.all(filled, axis=1)))

    # wells 計算
    left = np.concatenate(([np.inf], heights[:-1]))
    right = np.concatenate((heights[1:], [np.inf]))
    wells = float(np.sum(np.maximum(0.0, np.minimum(left, right) - heights)))

    extras = [holes, aggregate_height, bumpiness, max_height, complete_lines, wells]
    feats = flat + heights.tolist() + deltas.tolist() + extras
    return feats


# =============================
# 值網路（瘦身版，速度更快）
# =============================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, p: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(x)))
        h = self.drop(h)
        return self.norm(x + h)


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, widths=(256, 128, 64), p=0.05):
        super().__init__()
        dims = [in_dim] + list(widths)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU(), ResidualBlock(dims[i + 1], p)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# =============================
# AI Player（向量化評估所有合法動作）
# =============================
class AIPlayerNN:
    def __init__(self, device: Optional[torch.device] = None, in_dim: Optional[int] = None, model: Optional[nn.Module] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 預設特徵維度：格子(rows*cols) + heights(cols) + deltas(cols-1) + 6 個額外特徵
        default_in = config.rows * config.columns + config.columns + (config.columns - 1) + 6
        self.in_dim = in_dim or default_in
        self.model = model or ValueNet(self.in_dim).to(self.device)
        self.model.eval()

    # ---- 讓舊版載入介面可相容（若你有載入 .pt）----
    @classmethod
    def load(cls, model_path: str, device: Optional[torch.device] = None):
        inst = cls(device=device)
        sd = torch.load(model_path, map_location=inst.device)
        inst.model.load_state_dict(sd)
        inst.model.eval()
        return inst

    # ---- 列舉所有合法動作 ----
    def _enumerate_moves(self, shot, p) -> List[Tuple[int, int]]:
        moves: List[Tuple[int, int]] = []
        # 對所有 rotation & x 進行合法性檢查
        for rot in range(len(config.shapes[p.shape])):
            tmpl = copy.deepcopy(p)
            tmpl.rotation = rot
            for x in range(-1, config.columns):  # 搜尋範圍稍微放寬，避免卡邊界
                sim = copy.deepcopy(tmpl)
                sim.x = x
                if Handler.isValidPosition(shot, sim):
                    moves.append((x, rot))
        return moves

    # ---- 評估所有合法動作（單次 forward）----
    @torch.no_grad()
    def find_best_move(self, game_state, piece, gamma: float = 0.985) -> Optional[Tuple[int, int]]:
        legal_moves = self._enumerate_moves(game_state, piece)
        if not legal_moves:
            return None

        feats: List[List[float]] = []
        rewards: List[float] = []
        done_flags: List[bool] = []

        for (x, rot) in legal_moves:
            sim_shot = copy.deepcopy(game_state)
            sim_piece = copy.deepcopy(piece)
            sim_piece.x = x
            sim_piece.rotation = rot

            prev_score = sim_shot.score
            Handler.instantDrop(sim_shot, sim_piece)
            # 消行（此處只當 tie-breaker 獎勵，主體仍依賴 value）
            Handler.eliminateFilledRows(sim_shot, sim_piece)
            r = float(sim_shot.score - prev_score)

            feats.append(_extract_features_from_board_np(sim_shot.status))
            rewards.append(r)
            done_flags.append(False)

        inputs = torch.tensor(feats, dtype=torch.float32, device=self.device)
        vals = self.model(inputs).squeeze(-1).cpu().numpy()

        rewards_np = np.asarray(rewards, dtype=np.float32)
        done_np = np.asarray(done_flags, dtype=bool)
        q = rewards_np + gamma * vals * (~done_np)

        best_idx = int(q.argmax())
        return legal_moves[best_idx]

    # ---- 推論分數（在訓練外部也可能用得到）----
    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def predict_value(self, boards: List) -> np.ndarray:
        batch = torch.tensor([_extract_features_from_board_np(b) for b in boards],
                             dtype=torch.float32, device=self.device)
        v = self.model(batch).squeeze(-1).cpu().numpy()
        return v
