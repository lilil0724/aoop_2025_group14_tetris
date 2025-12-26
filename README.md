# Tetris Battle - 俄羅斯方塊對戰版

這是一個基於 Python 和 Pygame 開發的俄羅斯方塊對戰遊戲，支援單人、雙人區域網對戰以及與 AI 對戰。

## 1. 環境安裝

請確保您的電腦已安裝 Python 3.x。

### 使用虛擬環境安裝 (建議)

建議使用虛擬環境來隔離專案依賴，避免與系統其他套件衝突：

1.  **建立虛擬環境**：
    ```bash
    # Windows
    python -m venv venv

    # Linux / macOS
    python3 -m venv venv
    ```

2.  **啟動虛擬環境**：
    ```bash
    # Windows (PowerShell)
    .\venv\Scripts\Activate.ps1
    
    # Windows (CMD)
    .\venv\Scripts\activate.bat

    # Linux / macOS
    source venv/bin/activate
    ```

3.  **安裝依賴項**：
    啟動虛擬環境後，執行以下指令安裝 `requirements.txt` 中列出的所有函式庫：
    ```bash
    pip install -r requirements.txt
    ```

### 執行遊戲
啟動虛擬環境後，進入專案根目錄執行：

```bash
python tetris/main.py
```

---

## 2. 如何遊玩

遊戲支援多種模式，您可以透過主選單進行選擇：

*   **Solo Mode (單人模式)**：經典的單人練習模式。
*   **1v1 Local (雙人對戰)**：在同一台電腦上進行雙人對戰。
*   **1vAI Battle (AI 對戰)**：與不同難度的 AI 進行對戰。
*   **LAN Battle (區域網對戰)**：透過區域網路與其他玩家連線對戰。

### 預設控制鍵位 (可在 Settings 中修改)

| 動作 | 玩家 1 (P1) | 玩家 2 (P2) |
| :--- | :--- | :--- |
| **左移** | `A` | `Left Arrow` |
| **右移** | `D` | `Right Arrow` |
| **加速下落** | `S` | `Down Arrow` |
| **順時針旋轉** | `W` | `Up Arrow` |
| **逆時針旋轉** | `Q` | `L` |
| **硬降 (直接到底)** | `Left Shift` | `Right Shift` |
| **暫停** | `ESC` | `ESC` |

---

## 3. 遊戲規則

### 基礎規則
*   方塊會不斷從上方掉落，填滿一整行即可消除。
*   當方塊堆疊到頂部時，遊戲結束。

### 對戰機制 (PVP / PVE / LAN)
*   **垃圾行攻擊**：當您一次消除多行方塊時，會向對手發送「垃圾行」。
    *   消除 2 行：發送 1 行垃圾。
    *   消除 3 行：發送 2 行垃圾。
    *   消除 4 行 (Tetris)：發送 4 行垃圾。
*   **B2B (Back-to-Back)**：連續進行 Tetris 消除會獲得額外的攻擊加成。
*   **Perfect Clear**：全消（清空整個版面）會對對手造成毀滅性的打擊。

### AI 難度
*   **Weighted AI**：基於權衡演算法的 AI，反應較為人性化。
*   **Expert AI**：極速反應的專家級 AI，挑戰難度極高。
*   您可以在 AI 選擇選單中調整 **AI Speed** (Slow / Normal / Fast / Instant)。
...sss