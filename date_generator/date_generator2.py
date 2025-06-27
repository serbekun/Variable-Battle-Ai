import torch
import random
import json
import os
from tqdm import tqdm

DATE_GENERATE_GAME_PER = 1_000_000
DATA_SET_SAVE_PATH = "../date_packs/"
DATA_SET_NAME = "data_dg2_1.ndjson"
DATA_END_SAVE_PATH = os.path.join(DATA_SET_SAVE_PATH, DATA_SET_NAME)
BATCH_SIZE = 1024
MAX_ROUNDS = 100 

os.makedirs(DATA_SET_SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use device: {device}")

def choose_action(hp, enemy_hp, attack, heal, last_action=None):
    
    if hp < 30:
        if enemy_hp < 40:  
            weights = [40, 10, 10, 30, 10] 
        else:  
            weights = [10, 25, 50, 5, 10]

    elif hp > enemy_hp + 30:
        weights = [50, 15, 5, 25, 5]
    
    else:
        weights = [35, 20, 20, 15, 10]
    
    if last_action == 1:
        weights[1] = max(5, weights[1] - 15)
    
    return random.choices([0, 1, 2, 3, 4], weights=weights)[0]

def simulate_battle():

    human_hp = torch.randint(90, 160, (1,)).item()
    model_hp = torch.randint(90, 160, (1,)).item()
    human_attack = torch.randint(25, 85, (1,)).item()
    model_attack = torch.randint(25, 85, (1,)).item()
    human_heal = torch.randint(15, 45, (1,)).item()
    model_heal = torch.randint(15, 45, (1,)).item()
    
    round_count = 1
    last_human_action = None
    last_model_action = None
    battle_history = []
    
    while human_hp > 0 and model_hp > 0 and round_count <= MAX_ROUNDS:
        
        human_action = choose_action(
            human_hp, model_hp, human_attack, human_heal, last_human_action
        )
        model_action = choose_action(
            model_hp, human_hp, model_attack, model_heal, last_model_action
        )
        
        human_block = (human_action == 1)
        model_block = (model_action == 1)
        
        dmg_to_model = 0
        dmg_to_human = 0
        heal_to_human = 0
        heal_to_model = 0

        if human_action == 0:
            dmg_to_model = human_attack
        elif human_action == 2:
            heal_to_human = human_heal
        elif human_action == 3:
            dmg_to_model = int(human_attack * 1.8) if random.random() > 0.3 else 0
        elif human_action == 4:
            heal_to_human = int(human_heal * 1.5) if random.random() > 0.4 else 0
        
        if model_action == 0:
            dmg_to_human = model_attack
        elif model_action == 2:
            heal_to_model = model_heal
        elif model_action == 3:
            dmg_to_human = int(model_attack * 1.8) if random.random() > 0.3 else 0
        elif model_action == 4:
            heal_to_model = int(model_heal * 1.5) if random.random() > 0.4 else 0

        if model_block: 
            dmg_to_model = max(1, dmg_to_model // 2)
        if human_block: 
            dmg_to_human = max(1, dmg_to_human // 2)
        
        new_human_hp = max(0, human_hp + heal_to_human - dmg_to_human)
        new_model_hp = max(0, model_hp + heal_to_model - dmg_to_model)
        
        record = {
            "round": round_count,
            "human": {
                "hp": human_hp,
                "new_hp": new_human_hp,
                "attack": human_attack,
                "heal": human_heal,
                "block": human_block,
                "action": human_action,
                "damage_dealt": dmg_to_model,
                "healing_done": heal_to_human
            },
            "model": {
                "hp": model_hp,
                "new_hp": new_model_hp,
                "attack": model_attack,
                "heal": model_heal,
                "block": model_block,
                "action": model_action,
                "damage_dealt": dmg_to_human,
                "healing_done": heal_to_model
            }
        }
        
        battle_history.append(record)
        
        human_hp, model_hp = new_human_hp, new_model_hp
        last_human_action = human_action
        last_model_action = model_action
        round_count += 1
    
    result = "draw"
    if human_hp <= 0 and model_hp > 0:
        result = "model_win"
    elif model_hp <= 0 and human_hp > 0:
        result = "human_win"

    for record in battle_history:
        record["battle_result"] = result
        yield json.dumps(record, ensure_ascii=False)

def generate_dataset(target_records):
    with open(DATA_END_SAVE_PATH, "w", encoding="utf-8") as f:
        with tqdm(total=target_records, desc="Generating data") as pbar:
            records_generated = 0
            while records_generated < target_records:
                for record in simulate_battle():
                    f.write(record + "\n")
                    records_generated += 1
                    pbar.update(1)
                    if records_generated >= target_records:
                        break
    print("Generated data saved to", DATA_END_SAVE_PATH)

if __name__ == "__main__":
    generate_dataset(DATE_GENERATE_GAME_PER)