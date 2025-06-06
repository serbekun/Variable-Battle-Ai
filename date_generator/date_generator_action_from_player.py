import random
import json
import os

MODELNAME       = "model"
GAME_LOG_SAVE   = "../date_packs/data_fp_attack_train.ndjson"

def show_display(player_hp, player_attack, player_heal, player_block,
                 bot_hp, bot_attack, bot_heal, bot_block, round_count):
    print("turn -", round_count)
    print(f"Player hp - {player_hp} | Bot hp = {bot_hp}")
    print(f"Player attack = {player_attack} | Bot attack = {bot_attack}")
    print(f"Player heal = {player_heal} | Bot heal = {bot_heal}")
    print("block status", player_block)
    print("1. Attack\n2. Heal\n3. Block\n4. Increase attack\n5. Increase heal")

def append_record(record: dict):
    os.makedirs(os.path.dirname(GAME_LOG_SAVE), exist_ok=True)
    with open(GAME_LOG_SAVE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_action():
    round_count = 0
    while True:
        round_count += 1
        
        player_hp      = random.randint(1, 150)
        player_attack  = random.randint(1, 100)
        player_heal    = random.randint(1, 100)
        player_block   = bool(random.getrandbits(1))
        
        bot_hp         = random.randint(1, 150)
        bot_attack     = random.randint(1, 100)
        bot_heal       = random.randint(1, 100)
        bot_block      = bool(random.getrandbits(1))
        bot_action     = random.randint(0, 4)

        show_display(player_hp, player_attack, player_heal, player_block,
                     bot_hp, bot_attack, bot_heal, bot_block, round_count)
        player_action = input(": ").strip()
        if player_action.lower() == "exit":
            break

        record = {
            "round_count": round_count,
            "human": {
                "hp":     player_hp,
                "attack": player_attack,
                "heal":   player_heal,
                "block":  player_block,
                "action": int(player_action),
            },
            MODELNAME: {
                "hp":     bot_hp,
                "attack": bot_attack,
                "heal":   bot_heal,
                "block":  bot_block,
                "action": bot_action,
            }
        }

        append_record(record)

if __name__ == "__main__":
    print("data was successful save ti the file", GAME_LOG_SAVE)
    get_action()