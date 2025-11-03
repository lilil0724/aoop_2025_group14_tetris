import torch
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
import numpy as np

# --- 訓練參數 ---
EPISODES = 2000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 10 # 每隔多少個 episode 更新一次 target network

def main():
    # 初始化環境和 Agent
    env = TetrisEnv()
    state_size = len(env.reset())
    agent = DQNAgent(state_size=state_size)
    
    scores = []
    
    for e in range(EPISODES):
        current_state = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            steps += 1
            # 1. 獲取所有可能的下一步
            possible_states = env.get_possible_states()
            
            # 2. Agent 決定動作
            action = agent.act(possible_states)
            
            if action is None: # 沒有合法移動，遊戲結束
                break

            # 3. 在環境中執行動作
            next_state_properties, reward, done = env.step(action)
            score += reward
            
            # 獲取下一步的所有可能狀態，用於計算 target Q value
            next_possible_states = env.get_possible_states() if not done else {}
            
            # 4. 將經驗存入記憶體
            agent.remember(current_state, action, reward, next_possible_states, done)
            
            current_state = next_state_properties

            # 5. 從記憶體中學習
            loss = agent.replay(BATCH_SIZE)

            if done:
                break
        
        scores.append(score)
        avg_score = np.mean(scores[-100:]) # 最近100場的平均分
        
        print(f"Episode: {e+1}/{EPISODES}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}, Steps: {steps}")

        # 定期更新 Target Network
        if (e + 1) % TARGET_UPDATE_FREQUENCY == 0:
            agent._update_target_network()
            print(f"--- Target network updated at episode {e+1} ---")

        # 儲存模型
        if (e + 1) % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f"tetris_dqn_episode_{e+1}.pth")
            print(f"Model saved at episode {e+1}")


if __name__ == "__main__":
    main()
