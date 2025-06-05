import random
import json
import os

DATE_GENERATE_GAME_PER = 1000000
DATA_SET_SAVE_PATH = "../date_packs/"
DATA_SET_NAME = "data__dg2_2" + ".json"
DATA_END_SAVE_PATH = DATA_SET_SAVE_PATH + DATA_SET_NAME

def simulate_battle():
    records = []
    human_max_hp = random.randint(100, 150)
    model_max_hp = random.randint(100, 150)
    human_hp = human_max_hp
    model_hp = model_max_hp

    human_attack = random.randint(20, 100)
    model_attack = random.randint(20, 100)

    human_heal = random.randint(10, 50)
    model_heal = random.randint(10, 50)

    round_count = 1

    while human_hp > 0 and model_hp > 0:
        if human_hp < 0.3 * human_max_hp:
            human_action = random.choices([0, 1, 2, 3, 4], weights=[15, 20, 40, 10, 15])[0]
        else:
            human_action = random.choices([0, 1, 2, 3, 4], weights=[40, 20, 20, 10, 10])[0]

        if model_hp < 0.3 * model_max_hp:
            model_action = random.choices([0, 1, 2, 3, 4], weights=[15, 20, 40, 10, 15])[0]
        else:
            model_action = random.choices([0, 1, 2, 3, 4], weights=[40, 20, 20, 10, 10])[0]

        human_block = (human_action == 1)
        model_block = (model_action == 1)

        if human_action == 0:
            damage_to_model = human_attack
        elif human_action == 3:
            damage_to_model = int(human_attack * 1.5)
        else:
            damage_to_model = 0

        if model_action == 0:
            damage_to_human = model_attack
        elif model_action == 3:
            damage_to_human = int(model_attack * 1.5)
        else:
            damage_to_human = 0

        if model_block and damage_to_model > 0:
            damage_to_model = int(damage_to_model / 2)
        if human_block and damage_to_human > 0:
            damage_to_human = int(damage_to_human / 2)

        if human_action == 2:
            human_heal_amount = human_heal
        elif human_action == 4:
            human_heal_amount = int(human_heal * 1.5)
        else:
            human_heal_amount = 0

        if model_action == 2:
            model_heal_amount = model_heal
        elif model_action == 4:
            model_heal_amount = int(model_heal * 1.5)
        else:
            model_heal_amount = 0

        updated_human_hp = max(0, min(human_max_hp, human_hp - damage_to_human + human_heal_amount))
        updated_model_hp = max(0, min(model_max_hp, model_hp - damage_to_model + model_heal_amount))

        round_record = {
            "round_count": round_count,
            "human": {
                "hp": updated_human_hp,
                "attack": human_attack,
                "heal": human_heal,
                "block": human_block,
                "action": human_action
            },
            "model": {
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

        if human_hp == 0 or model_hp == 0:
            break

    return records

def generate_dataset(target_records):

    if os.path.exists(DATA_END_SAVE_PATH):
        with open(DATA_END_SAVE_PATH, "r") as json_file:
            try:
                dataset = json.load(json_file)
            except json.JSONDecodeError:
                dataset= []
    else:
        dataset = []

    dataset = []
    while len(dataset) < target_records:
        battle_records = simulate_battle()
        dataset.extend(battle_records)
    return dataset[:target_records]

if __name__ == "__main__":
    data = generate_dataset(DATE_GENERATE_GAME_PER)
    with open(DATA_END_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("generated 'meaningful_dataset.json'.")