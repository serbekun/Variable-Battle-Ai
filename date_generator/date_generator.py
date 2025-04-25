import random
import json
from tqdm import tqdm

def get_action(hp, opponent_hp, round_count, attack, heal):
    if hp < 50:
        return 2
    elif opponent_hp < 40:
        return 1
    elif round_count > 60:
        return 1
    elif round_count < 100:
        inc_heal_or_attack = random.randint(1, 2)
        if inc_heal_or_attack == 1 and attack < 50:
            return 4
        elif inc_heal_or_attack == 2 and heal < 50:
            return 5
        else:
            return inc_heal_or_attack
    else:
        return random.randint(1, 5)

def generate_session(num_rounds):
    session_data = []
    human_hp, vb_hp = 100, 100
    human_attack, human_heal = 10, 5
    vb_attack, vb_heal = 5, 5
    human_block, vb_block = False, False

    for round_count in range(num_rounds):
        human_action = get_action(human_hp, vb_hp, round_count, human_attack, human_heal)
        vb_action = get_action(vb_hp, human_hp, round_count, vb_attack, vb_heal)

        if human_action == 1:  
            if vb_block:
                vb_block = False
            else:
                vb_hp -= human_attack
        elif human_action == 2:
            human_hp += human_heal
        elif human_action == 3:
            human_block = True
        elif human_action == 4:
            human_attack += 5
        elif human_action == 5:
            human_heal += 5

        if vb_action == 1:
            if human_block:
                human_block = False
            else:
                human_hp -= vb_attack
        elif vb_action == 2:
            vb_hp += vb_heal
        elif vb_action == 3:
            vb_block = True
        elif vb_action == 4:
            vb_attack += 5
        elif vb_action == 5:
            vb_heal += 5

        session_data.append({
            "round_count": round_count,
            "human": {
                "hp": human_hp, "attack": human_attack, "heal": human_heal,
                "block": human_block, "action": str(human_action)
            },
            "vb_model1": {
                "hp": vb_hp, "attack": vb_attack, "heal": vb_heal,
                "block": vb_block, "action": vb_action
            }
        })

    return session_data

def generate_data(num_sessions, num_rounds_per_session):
    all_sessions = []
    for session in tqdm(range(num_sessions), desc="generation session"):
        session_data = generate_session(num_rounds_per_session)
        all_sessions.append(session_data)
    return all_sessions

def save_to_json(data, file_name):
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)

def main():
    game_data = generate_data(NUM_SESSIONS, NUM_ROUNDS_PER_SESSION)
    save_to_json(game_data, FILE_PATH)
    print(f"date successful save to'{FILE_PATH}'!")

NUM_SESSIONS = 10000
NUM_ROUNDS_PER_SESSION = 1000
FILE_NAME = "data_example.json"
FILE_PATH = "../date_packs/" +  FILE_NAME

if __name__ == "__main__":
    main()
