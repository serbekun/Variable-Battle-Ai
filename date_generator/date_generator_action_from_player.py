import random
import json
import os

MODELNAME = "model"
GAME_LOG_SAVE = "../date_packs/data_1_from_player.json"

def show_display(player_hp, player_attack, player_heal, player_block, bot_hp, bot_attack, bot_heal, bot_block, round_count):
    print("turn -", round_count)
    print(f"Player hp - {player_hp} | Bot hp = {bot_hp}")
    print(f"Player attack = {player_attack} | Bot attack = {bot_attack}")
    print(f"Player heal = {player_heal} | Bot heal = {bot_heal}")
    print("block status", player_block)
    print("1. Attack\n2. Heal\n3. Block\n4. Increase attack\n5. Increase heal")

def get_action():
    
    if os.path.exists(GAME_LOG_SAVE):
        with open(GAME_LOG_SAVE, "r") as json_file:
            try:
                session_data = json.load(json_file)
            except json.JSONDecodeError:
                session_data = []
    else:
        session_data = []

    
    while True:
    
        round_count = random.randint(1, 100)
        
        player_hp = random.randint(1, 150)
        player_attack = random.randint(1, 100)
        player_heal = random.randint(1, 100)
        player_block_set = random.randint(0, 1)
        if player_block_set == 0:
            player_block = False 
        else:
            player_block = True

        bot_hp = random.randint(1, 150)
        bot_attack = random.randint(1, 100)
        bot_heal = random.randint(1, 100)
        bot_block_set = random.randint(0, 1)
        if bot_block_set == 0:
            bot_block = False 
        else:
            bot_block = True
        bot_action = random.randint(0, 4)

        print("bot action -", bot_action)
        show_display(player_hp, player_attack, player_heal, player_block, bot_hp, bot_attack, bot_heal, bot_block, round_count)
        player_action = input(":")

        if player_action == "exit":
            exit(0)

        session_data.append({
            "round_count": round_count,
            "human": {
                "hp": player_hp, "attack": player_attack, "heal": player_heal,
                "block": player_block, "action": int(player_action)
            },
            MODELNAME: {
                "hp": bot_hp, "attack": bot_attack, "heal": bot_heal,
                "block": bot_block, "action": bot_action
            }
        })

        with open(GAME_LOG_SAVE, "w") as json_file:
            json.dump(session_data, json_file, indent=4)


get_action()