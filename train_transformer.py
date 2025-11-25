import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import TetrisDataset 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. 模型定義
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

class TetrisTransformer(nn.Module):
    def __init__(self, board_dim: int = 200, n_pieces: int = 7, d_model: int = 128, nhead: int = 4, num_layers: int = 3, action_dim: int = 64):
        super().__init__()
        self.board_proj = nn.Linear(board_dim, d_model)
        self.piece_emb = nn.Embedding(n_pieces, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, board_flat: torch.Tensor, piece_id: torch.Tensor) -> torch.Tensor:
        board_token = self.board_proj(board_flat)
        piece_token = self.piece_emb(piece_id)
        tokens = torch.stack([piece_token, board_token], dim=0)
        tokens = self.pos_encoder(tokens)
        output = self.transformer(tokens)
        cls_token = output[0]
        logits = self.action_head(cls_token)
        return logits

# -----------------------------
# 2. 訓練輔助函式
# -----------------------------
def collate_fn(batch):
    boards = []
    piece_ids = []
    action_ids = []
    for sample in batch:
        board = sample["board"]
        piece = sample["piece_id"]
        action = sample["action_id"]
        boards.append(board.reshape(-1))
        piece_ids.append(piece)
        action_ids.append(action)
    boards_t = torch.tensor(np.stack(boards), dtype=torch.float32)
    piece_ids_t = torch.tensor(piece_ids, dtype=torch.long)
    action_ids_t = torch.tensor(action_ids, dtype=torch.long)
    return boards_t, piece_ids_t, action_ids_t

# -----------------------------
# 3. 主訓練迴圈 (加入續練功能)
# -----------------------------
def train(
    dataset_path: str = "tetris_demo_data.npz",
    save_path: str = "transformer_tetris.pth",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    resume: bool = True # 新增開關：是否要載入舊模型
):
    print(f"🔥 開始訓練 Transformer | Device: {DEVICE}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ 錯誤：找不到資料集 {dataset_path}")
        return

    dataset = TetrisDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    print(f"📊 資料筆數: {len(dataset)}")

    # 建立模型
    model = TetrisTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --- 續練邏輯 ---
    start_epoch = 1
    if resume and os.path.exists(save_path):
        print(f"🔄 發現既有模型 {save_path}，正在載入以繼續訓練...")
        try:
            # 如果你有存 optimizer state 更好，這裡簡化只載入權重
            # 這樣 optimizer 的 momentum 會重置，但對微調影響不大
            model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            print("✅ 成功載入舊權重！")
        except Exception as e:
            print(f"⚠️ 載入失敗 ({e})，將重新開始訓練。")
    else:
        print("🆕 找不到舊模型或 resume=False，將從頭開始訓練。")

    # 訓練迴圈
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for boards, piece_ids, action_ids in dataloader:
            boards = boards.to(DEVICE)
            piece_ids = piece_ids.to(DEVICE)
            action_ids = action_ids.to(DEVICE)

            optimizer.zero_grad()
            logits = model(boards, piece_ids)
            loss = criterion(logits, action_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * boards.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == action_ids).sum().item()
            total_samples += boards.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples

        print(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已備份至 {save_path}")

    torch.save(model.state_dict(), save_path)
    print(f"🎉 訓練完成！最終模型: {save_path}")

if __name__ == "__main__":
    train(
        dataset_path="tetris_demo_data.npz", 
        epochs=500,        
        batch_size=256,
        lr=1e-4,
        resume=True # 設定為 True 即可續練
    )
