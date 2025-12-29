# Tetris Battle - 俄羅斯方塊對戰版

這是一個基於 Python 和 Pygame 開發的俄羅斯方塊對戰遊戲，支援單人、雙人區域網對戰以及與 AI 對戰。

## 1. 環境安裝

請確保您的電腦已安裝 Python 3.11。

### 使用虛擬環境安裝 

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

### 使用 Anaconda 建立 Conda 環境

⚠️ **請先前往 Anaconda 官方網站下載並安裝 Anaconda**  
👉 https://www.anaconda.com/download

安裝完成後，使用 **Anaconda Powershell Prompt** 建立並管理 conda 環境，之後使用 `conda activate` 即可啟動虛擬環境：

1.  **開啟 Anaconda Powershell Prompt**

2.  **建立 conda 環境**  
    > ⚠️ `tetris-battle` 為環境名稱，可依個人喜好自行命名
    ```bash
    conda create -n tetris-battle python=3.11 -y
    ```

3.  **啟動 conda 環境**：
    ```bash
    conda activate tetris-battle
    ```

4.  **安裝依賴項**：
    啟動 conda 環境後，執行以下指令安裝 `requirements.txt` 中列出的所有函式庫：
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

| 動作                | 玩家 1 (P1)  | 玩家 2 (P2)   |
| :------------------ | :----------- | :------------ |
| **左移**            | `A`          | `Left Arrow`  |
| **右移**            | `D`          | `Right Arrow` |
| **加速下落**        | `S`          | `Down Arrow`  |
| **順時針旋轉**      | `W`          | `Up Arrow`    |
| **逆時針旋轉**      | `Q`          | `L`           |
| **硬降 (直接到底)** | `Left Shift` | `Right Shift` |
| **暫停**            | `ESC`        | `ESC`         |

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
*   **Score Attack (分數對決)**：
    *   在對戰模式中，遊戲**不會**因為其中一方死亡而立即結束。
    *   遊戲會持續直到**所有玩家**都死亡為止。
    *   最終勝負將由**分數 (Score)** 決定，分數高者獲勝。這意味著即使您先死亡，若您的分數較高，仍有可能獲勝。

### AI 難度
*   **Weighted AI**：基於權衡演算法的 AI，反應較為人性化。
*   **Expert AI**：極速反應的專家級 AI，挑戰難度極高。
*   您可以在 AI 選擇選單中調整 **AI Speed** (Slow / Normal / Fast / Instant)。

---

## 4. 遊戲特色與技術細節

本專案實作了許多經典俄羅斯方塊的機制與進階功能：

*   **7-Bag Randomizer (7-Bag 發牌機制)**：
    為了確保遊戲公平性，我們實作了現代俄羅斯方塊標準的 7-Bag 發牌系統。這保證了每 7 個方塊中一定包含所有形狀（I, J, L, O, S, T, Z）各一次，避免了長時間不出長條 (I) 或特定方塊的情況。
*   **Ghost Piece (落點投影)**：
    遊戲支援顯示方塊落點預測 (Ghost)，幫助玩家更精準地預判方塊放置位置。此功能預設開啟，可在 `settings.py` 中調整。
*   **Advanced AI Algorithms**：
    *   **Weighted AI**：基於 **Dellacherie's Algorithm**，透過計算 "Landing Height", "Row Transitions", "Holes" 等特徵來評估最佳落點，表現穩定且人性化。
    *   **Neural Network AI (實驗性)**：專案包含一個基於 PyTorch 的 AI 模型架構 (`ai_player_nn.py`)，作為實驗性功能保留，未來可透過訓練模型來進一步強化 AI。
*   **解析度調整 (Resolution Scaling)**：
    *   支援多種解析度設定 (1280x720, 1024x576, 1600x900, 1920x1080)，適應不同螢幕尺寸（如筆記型電腦）。
    *   可在 **Settings** 選單中即時切換。

---

## 5. 專案結構

*   `tetris/`
    *   `main.py`: 遊戲程式入口，負責初始化與場景切換。
    *   `game_engine.py`: 核心遊戲迴圈，處理遊戲邏輯與狀態更新。
    *   `ai_weighted.py`: 基於 Dellacherie 演算法的權重式 AI。
    *   `ai_heuristic.py`: 啟發式 AI 邏輯。
    *   `ai_player_nn.py`: (實驗性) 基於類神經網路的 AI 模型。
    *   `network_utils.py`: 處理區域網路 (LAN) 連線與資料傳輸。
    *   `pieces.py`: 定義俄羅斯方塊的形狀與旋轉邏輯。
    *   `settings.py`: 全域設定檔 (按鍵綁定、音量、AI 速度)。
    *   `ui.py` & `menus.py`: 負責畫面繪製與選單介面。

---

## 6. 區域網對戰疑難排解

如果在使用 **LAN Battle** 模式時無法連線，請檢查以下事項：

1.  **防火牆設定**：請確保 Windows 防火牆允許 Python (或您的終端機) 通過，若連線失敗可嘗試暫時關閉防火牆測試。
2.  **同一網段**：確保兩台電腦連接到同一個 Wi-Fi 或路由器。
3.  **IP 位址選擇**：
    *   Host 端現在可以點擊 IP 輸入框來**切換不同的網路介面 IP**。
    *   若您使用 VPN (如 Hamachi, Radmin VPN) 或學校網路，請確保選擇對應的虛擬網卡 IP。
4.  **Port 設定**：
    *   預設 Port 為 `5555`。若該 Port 被佔用或被防火牆阻擋，雙方可約定修改為其他 Port (如 `12345`)。

---

## 7. 刪除環境

當您不再需要此專案的開發環境時，可依照下列方式刪除虛擬環境。

### 刪除 venv 虛擬環境

請先確認 **已離開虛擬環境**（若目前仍在虛擬環境中，請先執行 `deactivate`），再刪除 `venv` 資料夾：

```bash
# Windows
rmdir /s /q venv

# Linux / macOS
rm -rf venv
```

### 刪除 conda 環境

請先離開目前使用中的 conda 環境：

```bash
conda deactivate
```

接著刪除指定的 conda 環境（以下以 `tetris-battle` 為例）：

```bash
conda remove -n tetris-battle --all
```

tetris-battle 為環境名稱，若建立時使用不同名稱，請替換為實際的環境名稱。
