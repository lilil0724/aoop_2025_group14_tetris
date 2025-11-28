import pygame as pg
import sys
import config
from menus import main_menu, settings_menu, ai_selection_menu, lan_menu, game_over_screen
from game_engine import run_game

def main():
    pg.init()
    pg.font.init()
    
    screen = pg.display.set_mode((config.width, config.height))
    clock = pg.time.Clock()
    font = pg.font.SysFont(*config.font)
    
    current_mode = None
    
    while True:
        # 1 顯示主選單
        choice = main_menu(screen, font)
        
        if choice == "EXIT":
            pg.quit()
            sys.exit()
        elif choice == "SETTINGS":
            settings_menu(screen)
            continue
        
        # 2. 處理 AI 選擇邏輯
        selected_ai_mode = None # 預設無
        net_mgr = None # 預設無
        
        if choice == "PVE":
            # 如果選了 PVE，先跳出選擇 AI 難度的視窗
            ai_choice = ai_selection_menu(screen, font)
            if ai_choice == "BACK":
                continue # 放棄，回到主選單
            selected_ai_mode = ai_choice # 紀錄是 DQN 還是 HEURISTIC
            
        elif choice == "LAN":
            # 如果選了 LAN，跳出連線選單
            lan_mode, mgr = lan_menu(screen, font)
            if lan_mode is None:
                continue
            net_mgr = mgr
        
        current_mode = choice
        
        # 3. 進入遊戲
        while True:
            # 將 ai_mode 傳入 run_game
            result = run_game(screen, clock, font, current_mode, ai_mode=selected_ai_mode, net_mgr=net_mgr)
            
            if result == "MENU":
                if net_mgr: net_mgr.close()
                break # 回到主選單
            elif result == "RESTART":
                # LAN 模式下 Restart 比較複雜，這裡先簡單處理：斷線重連
                # 實際上應該發送 Restart 訊號，但為了簡化，LAN 模式下 Restart 回到選單
                if current_mode == 'LAN':
                    if net_mgr: net_mgr.close()
                    break
                continue # 重新開始這一局 (保持同樣的 AI 設定)
            
            if isinstance(result, tuple) and result[0] == "GAME_OVER":
                action = game_over_screen(screen, result[1])
                if action == "RESTART":
                    if current_mode == 'LAN':
                        if net_mgr: net_mgr.close()
                        break
                    continue
                elif action == "MENU":
                    if net_mgr: net_mgr.close()
                    break

if __name__ == "__main__":
    main()


 