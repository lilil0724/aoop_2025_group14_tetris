import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

DATA_FILE = "dataset.jsonl"
MODEL_FILE = "ai_model.pt"

# --- 讀取資料 ---
def load_dataset():
    X, y = [], []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            board = np.array(d["board"]) / 2.0  # 正規化 [0, 1]
            piece_id = ord(d["piece"][0]) / 90.0  # 把字母轉成小數
            next_piece_id = ord(d["next_piece"][0]) / 90.0
            feature = np.concatenate([board.flatten(), [piece_id, next_piece_id]])
            X.append(feature)
            # 動作 = rotation * 10 + x (共 4*10 = 40 類)
            y.append(d["target_rot"] * 10 + d["target_x"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# --- 模型定義 ---
class TetrisNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 主程式 ---
def train_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TetrisNet(X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(device)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        # 驗證
        model.eval()
        with torch.no_grad():
            test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Test Acc: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"✅ 模型已儲存至 {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
