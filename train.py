import cma
import numpy as np
import pickle
import multiprocessing as mp
import os
import time
from tetris_env import TetrisEnv

# --- 進化參數 ---
POPULATION_SIZE = 16   # 每一代有 16 個 AI 參賽
GENERATIONS = 100      # 總共進化 100 代
GAMES_PER_AGENT = 5    # 每個 AI 玩 5 場取平均 (減少運氣成分)
MAX_STEPS = 5000       # 每場最多玩幾步 (避免無限玩)

# 初始權重猜測 (根據文獻經驗):
# [總高度, 消行數, 空洞數, 粗糙度]
# 注意：CMA-ES 是求 "最小值"，所以我們要 "最大化分數" = "最小化負分"
# 我們希望：高度低(-), 消行多(+), 空洞少(-), 粗糙少(-)
# 初始種子：[-0.5, 0.76, -0.36, -0.18] (這是 Pierre Dellacherie 算法的變體)
# 修改 INITIAL_WEIGHTS
# Dellacherie 經驗值參考：
# Height: -1
# Row Trans: -1
# Col Trans: -1
# Holes: -4  (空洞懲罰最重)
# Wells: -1
INITIAL_WEIGHTS = [-1.37156088, -2.23096415, -0.74890419, -3.87641746, -0.53129402, -0.36264025,
  0.04413783, -0.91904935]
INITIAL_SIGMA = 0.1    # 突變幅度

# --- 評估函數 (Worker) ---
def evaluate_agent(weights):
    env = TetrisEnv()
    total_lines = 0
    tetris_count = 0
    for _ in range(GAMES_PER_AGENT):
        state = env.reset() # state 已經是 [agg_height, row_trans, col_trans, holes, wells]
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            steps += 1
            possible_next = env.get_possible_next_states()
            
            if not possible_next: break
            
            best_score = -float('inf')
            best_action = None
            
            for action, features in possible_next.items():
                # features 已經是 5 維向量
                # weights 也是 5 維向量
                score = np.dot(weights, features)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            if best_action:
                _, done = env.step(best_action)
                if env.last_cleared_lines == 4: # 需在 env 中記錄 last_cleared_lines
                    tetris_count += 1
                elif env.last_cleared_lines == 3:
                    tetris_count += 0.7
                elif env.last_cleared_lines == 2:
                    tetris_count += 0.3
            else:
                break
        
        # 我們優化目標是 "消行數"
        total_lines += env.line_count
        
    avg_lines = total_lines / GAMES_PER_AGENT
    score = avg_lines + (tetris_count * 10)
    # CMA-ES 求最小化，所以回傳負的消行數
    # 如果你是用 score 也可以，但 lines 比較直觀
    return -score, avg_lines 


# --- 主訓練迴圈 ---
def train_evolution():
    # 設定多進程
    num_workers = mp.cpu_count() - 3
    pool = mp.Pool(num_workers)
    
    # 初始化 CMA-ES
    es = cma.CMAEvolutionStrategy(INITIAL_WEIGHTS, INITIAL_SIGMA, {'popsize': POPULATION_SIZE})
    
    print(f"🧬 開始進化訓練... (Workers: {num_workers})")
    print(f"初始權重: {INITIAL_WEIGHTS}")
    
    best_ever_score = 0
    
    for gen in range(GENERATIONS):
        start_time = time.time()
        
        # 1. 生小孩 (Ask)
        solutions = es.ask()
        
        # 2. 考試 (Evaluate) - 平行處理
        # solutions 是一群權重向量
        results = pool.map(evaluate_agent, solutions)
        
        # 解包結果
        fitness_values = [r[0] for r in results] # 負分 (給 CMA-ES 用)
        lines_cleared = [r[1] for r in results]  # 實際消行數 (給人看)
        
        # 3. 更新家長 (Tell)
        es.tell(solutions, fitness_values)
        es.logger.add()
        
        # 4. 顯示進度
        current_best_score = -min(fitness_values)
        avg_gen_score = -np.mean(fitness_values)
        max_lines = max(lines_cleared)
        
        if current_best_score > best_ever_score:
            best_ever_score = current_best_score
            # 存檔
            best_weights = es.result.xbest
            with open("tetris_best_weights.pkl", "wb") as f:
                pickle.dump(best_weights, f)
            print(f"💾 新紀錄！權重已儲存。")
            
        print(f"Gen {gen+1} | Best: {current_best_score:.0f} | Avg: {avg_gen_score:.0f} | Max Lines: {max_lines:.1f} | Time: {time.time()-start_time:.1f}s")
        print(f"   Top Weights: {es.result.xbest}")
        
        es.disp()

    print("訓練結束！")
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Windows 必須
    mp.set_start_method('spawn', force=True)
    train_evolution()
