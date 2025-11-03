# train_segmented.py

import torch
import config
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import numpy as np
from collections import deque
import os

# --- 分段訓練計畫 ---
# 這是整個訓練的課程大綱。你可以自由調整每個階段的參數。
TRAINING_PHASES = [
    {
        "name": "Phase 1: Exploration",
        "episodes": 2000,
        "batch_size": 128,
        "learning_rate": 5e-4,  # 較高的學習率，快速學習
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,    # 探索率下降到 10%
        "epsilon_decay": 0.999
    },
    {
        "name": "Phase 2: Refinement",
        "episodes": 5000,
        "batch_size": 256,
        "learning_rate": 1e-4,  # 較低的學習率，穩定學習
        "epsilon_start": 0.1,   # 從上一階段的終點開始
        "epsilon_end": 0.02,   # 探索率進一步下降到 2%
        "epsilon_decay": 0.9995
    },
    {
        "name": "Phase 3: Fine-tuning",
        "episodes": 3000,
        "batch_size": 512,
        "learning_rate": 5e-5,  # 非常低的學習率，精細微調
        "epsilon_start": 0.02,  # 從上一階段的終點開始
        "epsilon_end": 0.001,  # 幾乎不再隨機探索
        "epsilon_decay": 0.9999
    }
]

def main():
    env = TetrisEnv()
    agent = DQNAgent(state_shape=(2, config.rows, config.columns))
    
    total_episodes_done = 0
    latest_model_path = None

    # --- 主迴圈：遍歷所有訓練階段 ---
    for i, phase in enumerate(TRAINING_PHASES):
        print(f"\n=============================================")
        print(f"  Starting: {phase['name']} (Phase {i+1}/{len(TRAINING_PHASES)})")
        print(f"=============================================")

        # 1. 如果不是第一階段，就加載上一階段訓練好的模型
        if latest_model_path and os.path.exists(latest_model_path):
            print(f"Loading model from previous phase: {latest_model_path}")
            agent.policy_net.load_state_dict(torch.load(latest_model_path))
            agent._update_target_network()

        # 2. 為新階段重設 Agent 的超參數
        agent.reconfigure_for_phase(phase)
        
        scores = deque(maxlen=100)
        
        # --- 內部迴圈：執行當前階段的訓練 ---
        for e in range(phase['episodes']):
            env.reset()
            done = False
            score = 0
            episode_steps = 0
            
            while not done:
                episode_steps += 1

                possible_next_states = env.get_possible_next_states()
                action = agent.act(possible_next_states)
                if action is None: break
                
                state_for_action = possible_next_states[action]
                _, reward, done = env.step(action)
                score += reward
                
                next_possible_states_after_move = env.get_possible_next_states() if not done else {}
                agent.remember(state_for_action, action, reward, next_possible_states_after_move, done)
                
                # 在記憶體池累積足夠經驗後才開始學習
                if len(agent.memory) > phase['batch_size'] * 4:
                    loss = agent.replay(phase['batch_size'])

                if done or episode_steps > 3000: break
            
            total_episodes_done += 1
            scores.append(score)
            avg_score = np.mean(scores)
            
            print(f"Total E: {total_episodes_done}, Phase E: {e+1}/{phase['episodes']}, Score: {score:.2f}, Avg: {avg_score:.2f}, Eps: {agent.epsilon:.4f}")

            if (e + 1) % 20 == 0:
                agent._update_target_network()

        # 3. 儲存當前階段的最終模型
        latest_model_path = f"tetris_transformer_phase{i+1}_final.pth"
        torch.save(agent.policy_net.state_dict(), latest_model_path)
        print(f"--- {phase['name']} complete. Model saved to {latest_model_path} ---")

if __name__ == "__main__":
    main()
