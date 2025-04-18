import random
import json

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

        # Обработка действий human
        if human_action == 1:  # атака
            if vb_block:
                vb_block = False
            else:
                vb_hp -= human_attack
        elif human_action == 2:  # лечение
            human_hp += human_heal
        elif human_action == 3:  # блок
            human_block = True
        elif human_action == 4:  # увеличение атаки
            human_attack += 5
        elif human_action == 5:  # увеличение лечения
            human_heal += 5

        # Обработка действий vb_model1
        if vb_action == 1:  # атака
            if human_block:
                human_block = False
            else:
                human_hp -= vb_attack
        elif vb_action == 2:  # лечение
            vb_hp += vb_heal
        elif vb_action == 3:  # блок
            vb_block = True
        elif vb_action == 4:  # увеличение атаки
            vb_attack += 5
        elif vb_action == 5:  # увеличение лечения
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
    for session in range(num_sessions):
        session_data = generate_session(num_rounds_per_session)
        all_sessions.append(session_data)
    return all_sessions

def save_to_json(data, file_name):
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)

def main():
    game_data = generate_data(NUM_SESSIONS, NUM_ROUNDS_PER_SESSION)
    save_to_json(game_data, FILE_PATH)
    print(f"Данные успешно сохранены в файл '{FILE_PATH}'!")

NUM_SESSIONS = 1000  # Количество сессий
NUM_ROUNDS_PER_SESSION = 100  # Количество раундов в одной сессии
FILE_NAME = "data_1.json"  # Название файла
FILE_PATH = "../date_packs/" +  FILE_NAME  # Путь к файлу

if __name__ == "__main__":
    main()
