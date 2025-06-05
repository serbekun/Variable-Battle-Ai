import torch
import random
import json
from tqdm import tqdm

# Количество записей (каждая запись — один раунд битвы)
DATE_GENERATE_GAME_PER = 10000000
DATA_SET_SAVE_PATH = "../date_packs/"
DATA_SET_NAME = "data__dg2_3.ndjson"  # изменили расширение для NDJSON
DATA_END_SAVE_PATH = DATA_SET_SAVE_PATH + DATA_SET_NAME
BATCH_SIZE = 1024

# init cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use device: {device}")

def simulate_battle():
    records = []
    
    human_max_hp = torch.randint(100, 150, (1,)).item()
    model_max_hp = torch.randint(100, 150, (1,)).item()
    human_hp = human_max_hp
    model_hp = model_max_hp

    human_attack = torch.randint(20, 100, (1,)).item()
    model_attack = torch.randint(20, 100, (1,)).item()

    human_heal = torch.randint(10, 50, (1,)).item()
    model_heal = torch.randint(10, 50, (1,)).item()

    round_count = 1

    while human_hp > 0 and model_hp > 0:
        human_action = random.choices([0, 1, 2, 3, 4], weights=[40, 20, 20, 10, 10])[0]
        model_action = random.choices([0, 1, 2, 3, 4], weights=[40, 20, 20, 10, 10])[0]

        human_block = (human_action == 1)
        model_block = (model_action == 1)

        damage_to_model = human_attack if human_action in [0, 3] else 0
        damage_to_human = model_attack if model_action in [0, 3] else 0

        damage_to_model = damage_to_model // 2 if model_block else damage_to_model
        damage_to_human = damage_to_human // 2 if human_block else damage_to_human

        updated_human_hp = max(0, human_hp - damage_to_human)
        updated_model_hp = max(0, model_hp - damage_to_model)

        round_record = {
            "round_count": round_count,
            "human": {
                "hp": updated_human_hp,
                "attack": human_attack,
                "heal": human_heal,
                "block": human_block,
                "action": human_action
            },
            "model": {  # ключ "model" соответствует MODEL_NAME_IN_JSON ниже
                "hp": updated_model_hp,
                "attack": model_attack,
                "heal": model_heal,
                "block": model_block,
                "action": model_action
            }
        }
        records.append(round_record)

        human_hp = updated_human_hp
        model_hp = updated_model_hp
        round_count += 1

    return records

def generate_dataset(target_records, batch_size=BATCH_SIZE):
    progress_bar = tqdm(total=target_records, desc="generated")
    count = 0
    with open(DATA_END_SAVE_PATH, "w", encoding="utf-8") as f:
        while count < target_records:
            battle_records = simulate_battle()
            # В формате NDJSON каждая запись записывается в отдельной строке
            for record in battle_records:
                json_record = json.dumps(record, ensure_ascii=False)
                f.write(json_record + "\n")
                count += 1
                progress_bar.update(1)
                if count >= target_records:
                    break
    progress_bar.close()

if __name__ == "__main__":
    generate_dataset(DATE_GENERATE_GAME_PER)
    print("generated data end save file was successfully saved")