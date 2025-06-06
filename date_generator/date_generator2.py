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

os.makedirs(DATA_SET_SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use device: {device}")

def simulate_battle():
    human_hp = torch.randint(100, 150, (1,)).item()
    model_hp = torch.randint(100, 150, (1,)).item()
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

        dmg_to_model = human_attack if human_action in [0, 3] else 0
        dmg_to_human = model_attack if model_action in [0, 3] else 0
        if model_block: dmg_to_model //= 2
        if human_block: dmg_to_human //= 2

        heal_to_human = human_heal if human_action == 2 else 0
        heal_to_model = model_heal if model_action == 2 else 0

        new_human_hp = max(0, human_hp + heal_to_human - dmg_to_human)
        new_model_hp = max(0, model_hp + heal_to_model - dmg_to_model)

        record = {
            "round_count": round_count,
            "human": {"hp": new_human_hp, "attack": human_attack, "heal": human_heal, "block": human_block, "action": human_action},
            "model": {"hp": new_model_hp, "attack": model_attack, "heal": model_heal, "block": model_block, "action": model_action}
        }

        yield json.dumps(record, ensure_ascii=False)
        human_hp, model_hp = new_human_hp, new_model_hp
        round_count += 1

def generate_dataset(target_records):
    with open(DATA_END_SAVE_PATH, "w", encoding="utf-8") as f:
        with tqdm(total=target_records, desc="Generating data") as pbar:
            for _ in range(target_records):
                for record in simulate_battle():
                    f.write(record + "\n")
                    pbar.update(1)
    print("Generated data saved to", DATA_END_SAVE_PATH)

if __name__ == "__main__":
    generate_dataset(DATE_GENERATE_GAME_PER)