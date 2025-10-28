# train.py (Multi-core / Parallel Processing Version)
import json
import random
import copy
import numpy as np
from tqdm import tqdm
import multiprocessing # <--- 核心模組：導入多處理函式庫

# 導入遊戲核心邏輯
import config
import pieces
import Handler
import shots
from ai_player_v2 import AIPlayer

# --- 遺傳演算法參數 ---
POPULATION_SIZE = 50
GENERATIONS = 30
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
GAMES_PER_AI = 5  # <-- 可以適當增加，讓評估更準確
MAX_MOVES = 500

WEIGHT_RANGES = {
    'lines_cleared': (0, 1),
    'aggregate_height': (-1, 0),
    'holes': (-1, 0),
    'bumpiness': (-1, 0)
}

def create_individual():
    """隨機創建一個個體 (一組權重)"""
    return {
        key: random.uniform(low, high)
        for key, (low, high) in WEIGHT_RANGES.items()
    }

# --- 核心改造 1：建立獨立的「測驗函式」 ---
def evaluate_fitness(individual):
    """
    評估單一個體的適應度。這個函式將被發送到不同的 CPU 核心上執行。
    """
    ai_player = AIPlayer(weights=individual)
    
    total_moves = 0
    for _ in range(GAMES_PER_AI):
        game_shot = shots.Shot()
        shape1 = random.choice(list(config.shapes.keys()))
        current_piece = pieces.Piece(5, 0, shape1)
        shape2 = random.choice(list(config.shapes.keys()))
        next_piece = pieces.Piece(5, 0, shape2)
        
        moves_made = 0
        for _ in range(MAX_MOVES):
            if Handler.isDefeat(game_shot, current_piece):
                break

            best_move = ai_player.find_best_move(game_shot, current_piece)
            if best_move is None:
                break

            current_piece.x, current_piece.rotation = best_move
            Handler.instantDrop(game_shot, current_piece)
            Handler.eliminateFilledRows(game_shot, current_piece)
            
            moves_made += 1
            
            shape3 = random.choice(list(config.shapes.keys()))
            current_piece, next_piece = next_piece, pieces.Piece(5, 0, shape3)
        total_moves += moves_made
        
    return total_moves / GAMES_PER_AI # 返回平均存活步數

def crossover(parent1, parent2):
    """將兩個親代的權重進行交叉，產生一個子代"""
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual):
    """對個體的權重進行隨機突變"""
    mutated_individual = copy.deepcopy(individual)
    for key in mutated_individual:
        if random.random() < MUTATION_RATE:
            change = random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
            mutated_individual[key] += change
            low, high = WEIGHT_RANGES[key]
            mutated_individual[key] = np.clip(mutated_individual[key], low, high)
    return mutated_individual

def main():
    """遺傳演算法主函式 (多核心版本)"""
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        print(f"\n===== 世代 {gen + 1}/{GENERATIONS} =====")
        
        # --- 核心改造 2：使用 Pool.map 進行平行評估 ---
        # Pool() 會自動偵測您電腦的 CPU 核心數並全部使用
        with multiprocessing.Pool() as pool:
            # 使用 tqdm 顯示進度條
            # pool.imap 的 "i" 代表 iterator，它能讓 tqdm 在任務完成時逐步更新
            fitness_scores = list(tqdm(pool.imap(evaluate_fitness, population), total=POPULATION_SIZE, desc="評估適應度"))

        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        
        best_individual, best_fitness = sorted_population[0]
        print(f"本世代最佳適應度 (平均存活步數): {best_fitness:.2f}")
        print(f"本世代最佳權重: {json.dumps(best_individual, indent=2)}")

        next_generation = []
        
        elite_count = POPULATION_SIZE // 10
        elites = [ind for ind, score in sorted_population[:elite_count]]
        next_generation.extend(elites)
        
        parents = [ind for ind, score in sorted_population]
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(parents[:POPULATION_SIZE // 2])
            parent2 = random.choice(parents[:POPULATION_SIZE // 2])
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            next_generation.append(mutated_child)
            
        population = next_generation

    print("\n===== 訓練完成 =====")
    # ... (最終評估和儲存部分保持不變) ...
    print("正在進行最終的嚴格評估...")
    with multiprocessing.Pool() as pool:
        final_fitness_scores = list(tqdm(pool.imap(evaluate_fitness, population), total=POPULATION_SIZE, desc="最終評估"))
    
    best_final_individual, best_score = sorted(zip(population, final_fitness_scores), key=lambda x: x[1], reverse=True)[0]
    
    print(f"\n訓練出的最終冠軍 AI 平均存活步數: {best_score:.2f}")
    print("訓練出的最佳權重為:")
    print(json.dumps(best_final_individual, indent=4))

    output_path = 'trained_weights.json'
    with open(output_path, 'w') as f:
        json.dump(best_final_individual, f, indent=4)
    print(f"\n最佳權重已儲存至 {output_path}")

# --- 核心改造 3：使用 if __name__ == '__main__': 保護主程式 ---
if __name__ == '__main__':
    # 這行可以確保在 Windows 等系統下，子處理程序不會錯誤地重複執行 main()
    multiprocessing.freeze_support() 
    main()

