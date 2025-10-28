# train.py
import json
import random
import copy
import numpy as np
from tqdm import tqdm  # 用於顯示進度條，需要 pip install tqdm

# 導入遊戲核心邏輯
import config
import pieces
import Handler
import shots
from ai_player_v2 import AIPlayer

# --- 遺傳演算法參數 ---
POPULATION_SIZE = 50      # 每一代的個體數量 (AI數量)
GENERATIONS = 30          # 總共要演化的世代數
MUTATION_RATE = 0.1       # 基因突變的機率
MUTATION_STRENGTH = 0.2   # 突變的強度
GAMES_PER_AI = 3          # 每個 AI 要玩幾場遊戲來評估適應度
MAX_MOVES = 500           # 每場遊戲的最大步數，防止無限循環

# 權重值的範圍
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

def run_simulation(ai_player):
    """
    運行一場無圖形介面的遊戲來評估 AI 的表現。
    返回這場遊戲的適應度分數 (修正為：總共放置的方塊數，代表存活能力)。
    """
    game_shot = shots.Shot()
    shape1 = random.choice(list(config.shapes.keys()))
    current_piece = pieces.Piece(5, 0, shape1)
    shape2 = random.choice(list(config.shapes.keys()))
    next_piece = pieces.Piece(5, 0, shape2)
    
    moves_made = 0 # <--- 新增：用來計算存活步數
    
    for _ in range(MAX_MOVES):
        if Handler.isDefeat(game_shot, current_piece):
            break

        best_move = ai_player.find_best_move(game_shot, current_piece)
        if best_move is None:
            break

        current_piece.x, current_piece.rotation = best_move
        Handler.instantDrop(game_shot, current_piece)
        Handler.eliminateFilledRows(game_shot, current_piece)
        
        moves_made += 1 # <--- 每成功放置一個方塊，計數器+1
        
        shape3 = random.choice(list(config.shapes.keys()))
        current_piece, next_piece = next_piece, pieces.Piece(5, 0, shape3)

    return moves_made # <--- 修正：返回存活的步數

def crossover(parent1, parent2):
    """將兩個親代的權重進行交叉，產生一個子代"""
    child = {}
    for key in parent1:
        # 隨機選擇繼承父親或母親的基因
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual):
    """對個體的權重進行隨機突變"""
    mutated_individual = copy.deepcopy(individual)
    for key in mutated_individual:
        if random.random() < MUTATION_RATE:
            change = random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
            mutated_individual[key] += change
            # 確保突變後的值仍在合理範圍內
            low, high = WEIGHT_RANGES[key]
            mutated_individual[key] = np.clip(mutated_individual[key], low, high)
    return mutated_individual

def main():
    """遺傳演算法主函式"""
    # 1. 初始化第一代族群
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        print(f"\n===== 世代 {gen + 1}/{GENERATIONS} =====")
        
        # 2. 評估每個個體的適應度
        fitness_scores = []
        for i in tqdm(range(POPULATION_SIZE), desc="評估適應度"):
            individual = population[i]
            ai_player = AIPlayer(weights=individual) # 直接在創建時植入獨立大腦
            
            scores = [run_simulation(ai_player) for _ in range(GAMES_PER_AI)]
            avg_score = sum(scores) / len(scores)
            fitness_scores.append(avg_score)

        # 3. 選擇、交叉、突變以產生新一代
        
        # 將族群和分數打包並排序
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        
        best_individual, best_fitness = sorted_population[0]
        print(f"本世代最佳適應度 (平均存活步數): {best_fitness:.2f}")
        print(f"本世代最佳權重: {json.dumps(best_individual, indent=2)}")

        next_generation = []
        
        # 保留前 10% 的菁英直接進入下一代 (菁英選擇)
        elite_count = POPULATION_SIZE // 10
        elites = [ind for ind, score in sorted_population[:elite_count]]
        next_generation.extend(elites)
        
        # 產生剩餘的 90%
        parents = [ind for ind, score in sorted_population]
        while len(next_generation) < POPULATION_SIZE:
            # 從表現較好的前半段中選擇父母
            parent1 = random.choice(parents[:POPULATION_SIZE // 2])
            parent2 = random.choice(parents[:POPULATION_SIZE // 2])
            
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            next_generation.append(mutated_child)
            
        population = next_generation

    # 訓練結束後，找出最終的冠軍
    print("\n===== 訓練完成 =====")
    final_fitness_scores = []
    for individual in tqdm(population, desc="最終評估"):
        ai_player = AIPlayer()
        ai_player.weights = individual
        scores = [run_simulation(ai_player) for _ in range(GAMES_PER_AI * 2)] # 更嚴格的最終測試
        final_fitness_scores.append(sum(scores) / len(scores))

    best_final_individual = sorted(zip(population, final_fitness_scores), key=lambda x: x[1], reverse=True)[0][0]
    
    print("訓練出的最佳權重為:")
    print(json.dumps(best_final_individual, indent=4))

    # 將最佳權重儲存到檔案
    output_path = 'trained_weights.json'
    with open(output_path, 'w') as f:
        json.dump(best_final_individual, f, indent=4)
    print(f"\n最佳權重已儲存至 {output_path}")


if __name__ == '__main__':
    main()

