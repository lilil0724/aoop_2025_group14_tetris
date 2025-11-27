import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import math
import os
import config
import tetris_env

# -------------------------------------------------
# 0. Transformer 模型定義
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
        return cls_token

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 1. 定義 Gym 環境
# -------------------------------------------------
class TetrisGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = tetris_env.TetrisEnv()
        self.observation_space = spaces.Box(low=0, high=7, shape=(201,), dtype=np.float32)
        self.action_space = spaces.Discrete(64) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        return self._get_obs(), {}

    def step(self, action_id):
        max_rot = 4
        min_x = -2
        max_x = config.columns + 3
        num_x = max_x - min_x + 1
        
        rot = action_id // num_x
        x_idx = action_id % num_x
        x = x_idx + min_x
        
        original_reward, game_over = self.env.step((x, rot))
        rl_reward = original_reward

        # 獎勵正規化邏輯
        if game_over:
            rl_reward = -1.0 
        else:
            rl_reward = 0.01
            if original_reward > 0:
                 rl_reward += original_reward / 100.0
            
        return self._get_obs(), rl_reward, game_over, False, {}

    def _get_obs(self):
        board_np = (self.env.board == 2).astype(np.float32).flatten()
        shape_list = list(config.shapes.keys())
        try:
            piece_id = shape_list.index(self.env.current_piece.shape)
        except:
            piece_id = 0
        obs = np.concatenate(([piece_id], board_np))
        return obs

# -------------------------------------------------
# 2. 特徵提取器
# -------------------------------------------------
class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        self.transformer = TetrisTransformer(
            board_dim=200, n_pieces=7, d_model=128, 
            nhead=4, num_layers=3, action_dim=64
        )
        
        pretrained_path = "transformer_tetris.pth"
        if os.path.exists(pretrained_path):
            try:
                print(f"🔄 正在載入預訓練權重: {pretrained_path} ...")
                pretrained_dict = torch.load(pretrained_path, map_location=DEVICE)
                model_dict = self.transformer.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'action_head' not in k}
                model_dict.update(pretrained_dict)
                self.transformer.load_state_dict(model_dict)
                print("✅ 成功載入 Transformer 權重！")
                
                # [初始狀態]：先凍結所有權重
                for param in self.transformer.parameters():
                    param.requires_grad = False
                print("❄️ Transformer 權重已初始化為凍結狀態 (等待 100k 步後解凍)")
                
            except Exception as e:
                print(f"⚠️ 權重載入失敗: {e}")
        else:
            print(f"⚠️ 找不到預訓練權重，將從頭訓練。")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        piece_id = observations[:, 0].long()
        board_flat = observations[:, 1:]
        
        board_token = self.transformer.board_proj(board_flat)
        piece_token = self.transformer.piece_emb(piece_id)
        
        tokens = torch.stack([piece_token, board_token], dim=0)
        tokens = self.transformer.pos_encoder(tokens)
        
        output = self.transformer.transformer(tokens)
        cls_token = output[0]
        
        return cls_token

# -------------------------------------------------
# 3. 客製化 Callback: 自動解凍權重
# -------------------------------------------------
class FreezeCallback(BaseCallback):
    def __init__(self, unfreeze_steps: int = 100000, verbose: int = 1):
        super().__init__(verbose)
        self.unfreeze_steps = unfreeze_steps
        self.is_unfrozen = False

    def _on_step(self) -> bool:
        # 檢查是否達到解凍步數
        if self.num_timesteps > self.unfreeze_steps and not self.is_unfrozen:
            print(f"\n🔓 達到 {self.num_timesteps} 步！正在解凍 Transformer 權重...")
            
            # 取得 Policy 中的特徵提取器 (Transformer)
            features_extractor = self.model.policy.features_extractor
            
            # 解凍權重：只要把開關打開，Optimizer 下一次就會自動更新它們
            for param in features_extractor.transformer.parameters():
                param.requires_grad = True
            
            # 這裡不需要再呼叫 optimizer.add_param_group，因為 SB3 初始化時已經把所有參數都傳給 optimizer 了
            
            self.is_unfrozen = True
            print("✅ Transformer 權重已解凍，現在開始會參與梯度更新！")
            
        return True
# -------------------------------------------------
# 4. 主訓練流程
# -------------------------------------------------
def train_rl(continue_training=False):
    print(f"🔥 啟動 RL 強化學習訓練 | Device: {DEVICE}")
    
    env = TetrisGymEnv()
    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./rl_checkpoints/", name_prefix="ppo_tetris")
    
    # [新增] 註冊 FreezeCallback，設定 100,000 步解凍
    freeze_callback = FreezeCallback(unfreeze_steps=100000)
    
    # 組合 Callbacks (Checkpoint + Freeze)
    callbacks = [checkpoint_callback, freeze_callback]
    
    model_path = "ppo_transformer_tetris_continued.zip"
    
    if continue_training and os.path.exists(model_path):
        print(f"🔄 載入 {model_path} 繼續訓練...")
        model = PPO.load(model_path, env=env, device=DEVICE)
        # 續練時，如果已經超過 10萬步，記得手動解凍 (或是 FreezeCallback 邏輯也會自動處理)
    else:
        print("✨ 開始全新的 PPO 訓練 (前 100k 步凍結特徵層)")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            device=DEVICE,
            policy_kwargs={
                "features_extractor_class": TransformerExtractor,
                "features_extractor_kwargs": {"features_dim": 128}, 
                "net_arch": dict(pi=[64, 64], vf=[64, 64])
            },
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01,
            clip_range=0.2,
        )
    
    print("🚀 開始訓練 (Target: 1M steps)...")
    try:
        model.learn(total_timesteps=1000000, callback=callbacks, reset_num_timesteps=not continue_training)
    except KeyboardInterrupt:
        print("🛑 訓練被手動中斷")
    
    model.save("ppo_transformer_tetris_continued")
    print("💾 模型已儲存為 ppo_transformer_tetris_continued.zip")

if __name__ == "__main__":
    # 建議設為 False 重新開始，以觀察完整的凍結->解凍過程
    train_rl(continue_training=False)
