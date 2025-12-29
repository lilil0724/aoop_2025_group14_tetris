import pygame as pg
import sys
import os
import config
# [恢復] 匯入 ai_selection_menu
from menus import main_menu, settings_menu, lan_menu, game_over_screen, ai_selection_menu
from game_engine import run_game

import settings

def main():
    pg.init()
    pg.font.init()
    
    # 初始化音效
    sounds = {}
    try:
        pg.mixer.init()
        base_path = os.path.dirname(os.path.abspath(__file__))
        clear_path = os.path.join(base_path, 'normal.mp3')
        tetris_path = os.path.join(base_path, 'tetris.mp3')
        
        if os.path.exists(clear_path):
            sounds['clear'] = pg.mixer.Sound(clear_path)
        if os.path.exists(tetris_path):
            sounds['tetris'] = pg.mixer.Sound(tetris_path)
            
        # Load and play BGM
        bgm_path = os.path.join(base_path, 'tetris_menu.mp3')
        game_bgm_path = os.path.join(base_path, 'tetris_game.mp3') # Game BGM path
        result_bgm_path = os.path.join(base_path, 'tetris_result.mp3') # Result BGM path

        if os.path.exists(bgm_path):
            pg.mixer.music.load(bgm_path)
            pg.mixer.music.set_volume(settings.VOLUME) # Set volume from settings
            pg.mixer.music.play(-1) # Loop indefinitely
            print(f"BGM loaded: {bgm_path}")
        else:
            print(f"BGM not found: {bgm_path}")
            
    except Exception as e:
        print(f"Warning: Sound initialization failed: {e}")

    # Remove RESIZABLE flag to prevent window maximization issues
    screen = pg.display.set_mode((config.width, config.height))
    pg.display.set_caption("Tetris Battle") # Ensure title is set
    
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
        
        # Switch to Game BGM
        if os.path.exists(game_bgm_path):
            try:
                pg.mixer.music.load(game_bgm_path)
                pg.mixer.music.set_volume(settings.VOLUME)
                pg.mixer.music.play(-1)
            except Exception as e:
                print(f"Error loading game BGM: {e}")

        # 3. 進入遊戲
        while True:
            # 將 ai_mode 傳入 run_game
            result = run_game(screen, clock, font, current_mode, ai_mode=selected_ai_mode, net_mgr=net_mgr, sounds=sounds)
            
            if result == "MENU":
                if net_mgr: net_mgr.close()
                break
            elif result == "RESTART":
                if current_mode == 'LAN':
                    if net_mgr: net_mgr.close()
                    break
                continue 
            
            if isinstance(result, tuple) and result[0] == "GAME_OVER":
                # Play Result BGM
                if os.path.exists(result_bgm_path):
                    try:
                        pg.mixer.music.stop()
                        pg.mixer.music.load(result_bgm_path)
                        pg.mixer.music.set_volume(settings.VOLUME)
                        pg.mixer.music.play(-1)
                    except Exception as e:
                        print(f"Error loading result BGM: {e}")

                action = game_over_screen(screen, result[1], net_mgr=net_mgr)
                if action == "RESTART":
                    if current_mode == 'LAN':
                        if net_mgr:
                            if net_mgr.is_server:
                                net_mgr.restart_game()
                            
                            # Reset the flag for the next game
                            net_mgr.restart_requested = False
                            
                            # Do NOT close net_mgr, just continue loop
                    
                    # Switch back to Game BGM for restart
                    if os.path.exists(game_bgm_path):
                        try:
                            pg.mixer.music.stop()
                            pg.mixer.music.load(game_bgm_path)
                            pg.mixer.music.set_volume(settings.VOLUME)
                            pg.mixer.music.play(-1)
                        except Exception as e:
                            print(f"Error loading game BGM: {e}")
                            
                    continue
                elif action == "MENU":
                    if net_mgr: net_mgr.close()
                    break
        
        # Switch back to Menu BGM
        pg.mixer.music.stop() # Stop current music
        if os.path.exists(bgm_path):
            try:
                pg.mixer.music.load(bgm_path)
                pg.mixer.music.set_volume(settings.VOLUME)
                pg.mixer.music.play(-1)
            except Exception as e:
                print(f"Error loading menu BGM: {e}")

if __name__ == "__main__":
    main()