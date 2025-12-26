import pygame as pg
import sys
import config
# [恢復] 匯入 ai_selection_menu
from menus import main_menu, settings_menu, lan_menu, game_over_screen, ai_selection_menu
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
        
        selected_ai_mode = None
        net_mgr = None
        
        if choice == "PVE":
            ai_choice = ai_selection_menu(screen, font)
            if ai_choice == "BACK":
                continue
            selected_ai_mode = ai_choice
            
        elif choice == "LAN":
            lan_mode, mgr = lan_menu(screen, font)
            if lan_mode is None:
                continue
            net_mgr = mgr
        
        current_mode = choice
        
        # 3. 進入遊戲
        while True:
            result = run_game(screen, clock, font, current_mode, ai_mode=selected_ai_mode, net_mgr=net_mgr)
            
            if result == "MENU":
                if net_mgr: net_mgr.close()
                break
            elif result == "RESTART":
                if current_mode == 'LAN':
                    if net_mgr: net_mgr.close()
                    break
                continue 
            
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