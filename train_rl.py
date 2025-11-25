import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
import math
import os

import config
import tetris_env
import copy

# -------------------------------------------------
# 0. Transformer 模型定義 (必須與訓練時完全一致)
# -------------------------------------------------

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 1. 定義 Gym 環境
# -------------------------------------------------
class TetrisGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = tetris_env.TetrisEnv()
        
        # 觀察空間: [piece_id (1) + board (200)] = 201
        self.observation_space = spaces.Box(low=0, high=7, shape=(201,), dtype=np.float32)
        
        # 動作空間: 64 個離散動作
        self.action_space = spaces.Discrete(64) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        return self._get_obs(), {}

    def step(self, action_id):
        # 解碼動作 (Action ID -> x, rot)
        max_rot = 4
        min_x = -2
        max_x = config.columns + 3
        num_x = max_x - min_x + 1
        
        rot = action_id // num_x
        x_idx = action_id % num_x
        x = x_idx + min_x
        
        # 執行動作
        # 注意：這裡我們依賴 tetris_env 內部的 step
        # 如果 env.step 回傳的 reward 已經包含消行獎勵，那很好
        original_reward, game_over = self.env.step((x, rot))
        
        rl_reward = original_reward
        
        # [強化獎勵機制]
        if game_over:
            rl_reward = -100.0  # 死亡重罰
        else:
            # 生存獎勵 (鼓勵活下去)
            rl_reward += 0.5
            
            # 我們可以額外獎勵消行 (如果 original_reward 已經有，這行可以省略)
            # 假設 env.line_count 會在 step 後更新
            # rl_reward += self.env.last_cleared_lines * 10.0 
            
        # 截斷 (Truncated): 這裡暫時不使用步數截斷，讓它自然死亡
        truncated = False
        
        return self._get_obs(), rl_reward, game_over, truncated, {}

    def _get_obs(self):
        # 取得盤面 (200維)
        board_np = (self.env.board == 2).astype(np.float32).flatten()
        
        # 取得方塊 ID
        shape_list = list(config.shapes.keys())
        piece_id = shape_list.index(self.env.current_piece.shape)
        
        # 拼接
        obs = np.concatenate(([piece_id], board_np))
        return obs

# -------------------------------------------------
# 2. 特徵提取器 (載入預訓練權重)
# -------------------------------------------------
class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # 建立 Transformer
        self.transformer = TetrisTransformer(
            board_dim=200, n_pieces=7, d_model=128, 
            nhead=4, num_layers=3, action_dim=64
        )
        
        # 載入預訓練權重
        pretrained_path = "transformer_tetris.pth"
        if os.path.exists(pretrained_path):
            try:
                print(f"🔄 正在載入預訓練權重: {pretrained_path} ...")
                pretrained_dict = torch.load(pretrained_path, map_location=DEVICE)
                
                # 過濾掉 action_head (因為 PPO 會自己建立新的 Policy Head)
                model_dict = self.transformer.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'action_head' not in k}
                
                model_dict.update(pretrained_dict)
                self.transformer.load_state_dict(model_dict)
                print("✅ 成功載入 Transformer 特徵提取層！(Transfer Learning)")
                
                # 可選：凍結 Transformer 權重，只訓練 Policy Head (先練手腳)
                # for param in self.transformer.parameters():
                #     param.requires_grad = False
                # print("❄️ Transformer 權重已凍結")
                
            except Exception as e:
                print(f"⚠️ 權重載入失敗: {e}，將從頭訓練。")
        else:
            print("⚠️ 找不到預訓練權重，將從頭訓練。")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: [Batch, 201]
        piece_id = observations[:, 0].long()
        board_flat = observations[:, 1:]
        
        # 手動執行 Transformer 前半段
        board_token = self.transformer.board_proj(board_flat)
        piece_token = self.transformer.piece_emb(piece_id)
        
        tokens = torch.stack([piece_token, board_token], dim=0)
        tokens = self.transformer.pos_encoder(tokens)
        
        output = self.transformer.transformer(tokens)
        cls_token = output[0] # [Batch, 128]
        
        return cls_token

# -------------------------------------------------
# 3. 主訓練流程
# -------------------------------------------------
def train_rl():
    print(f"🔥 啟動 RL 強化學習訓練 | Device: {DEVICE}")
    
    # 建立環境
    env = TetrisGymEnv()
    
    # 定義 Checkpoint (每 50000 步存一次)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./rl_checkpoints/",
        name_prefix="ppo_tetris"
    )
    
    # PPO 設定
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        device=DEVICE,
        policy_kwargs={
            "features_extractor_class": TransformerExtractor,
            "features_extractor_kwargs": {"features_dim": 128}, 
            "net_arch": dict(pi=[64, 64], vf=[64, 64]) # Policy Head & Value Head
        },
        learning_rate=1e-5, # 降低學習率，保護預訓練權重
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,      # 增加熵，鼓勵探索
    )
    
    print("🚀 開始訓練 (Target: 1M steps)...")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("🛑 訓練被手動中斷")
    
    # 最終存檔
    model.save("ppo_transformer_tetris_final")
    print("💾 最終 RL 模型已儲存為 ppo_transformer_tetris_final.zip")

if __name__ == "__main__":
    # 如果你想從頭練，就呼叫 train_rl()
    # train_rl()
    
    # 如果你想接續練，就用這段：
    model_path = "ppo_transformer_tetris_final.zip" # 上次存的檔
    if os.path.exists(model_path):
        print(f"🔄 載入 {model_path} 繼續訓練...")
        env = TetrisGymEnv()
        model = PPO.load(model_path, env=env, device=DEVICE)
        model.learn(total_timesteps=100000, reset_num_timesteps=False)
        model.save("ppo_transformer_tetris_continued")
        print("💾 續練完成並存檔")
    else:
        print("❌ 找不到舊檔，開始新訓練")
        train_rl()