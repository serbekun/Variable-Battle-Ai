import torch
import torch.nn as nn
import numpy as np
import random
import os
from tqdm import tqdm

# Параметры
INPUT_SIZE = 9
HIDDEN_SIZE = 126
OUTPUT_SIZE = 5
MODEL_NAME = "vb_model1"
MODEL_PATH = "../models/" + MODEL_NAME + ".pth"
LOSSLOGSAVEPATH = "../models_loss_log/" + MODEL_NAME + ".json"

NUM_GAMES = 10000000

def get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal):
    if bot_hp < 50:
        return 2
    elif player_hp < 40:
        return 1
    elif round_count > 60:
        return 1
    elif round_count < 100:
        inc = random.randint(1, 2)
        if inc == 1 and bot_attack < 50:
            return 4
        elif inc == 2 and bot_heal < 50:
            return 5
        else:
            return inc
    else:
        return random.randint(1, 5)

def generate_game(num_rounds):
    game_data = []
    player_hp, bot_hp = 100, 100
    player_attack, player_heal = 5, 5
    bot_attack, bot_heal = 5, 5
    player_block, bot_block = False, False

    for round_count in range(num_rounds):
        player_action = get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal)
        bot_action = get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal)

        if player_action == 1:
            if bot_block:
                bot_block = False
            else:
                bot_hp -= player_attack
        elif player_action == 2:
            player_hp += player_heal
        elif player_action == 3:
            player_block = True
        elif player_action == 4:
            player_attack += 5
        elif player_action == 5:
            player_heal += 5

        if bot_action == 1:
            if player_block:
                player_block = False
            else:
                player_hp -= bot_attack
        elif bot_action == 2:
            bot_hp += bot_heal
        elif bot_action == 4:
            bot_attack += 5
        elif bot_action == 5:
            bot_heal += 5

        game_data.append({
            "round_count": round_count,
            "player": {
                "hp": player_hp, "attack": player_attack, "heal": player_heal,
                "block": player_block, "action": player_action
            },
            "bot": {
                "hp": bot_hp, "attack": bot_attack, "heal": bot_heal,
                "block": bot_block, "action": bot_action
            }
        })

    return game_data

def prepare_data(data):
    X, y = [], []
    for entry in data:
        features = [
            entry["round_count"],
            entry["player"]["hp"], entry["player"]["attack"], entry["player"]["heal"], int(entry["player"]["block"]),
            entry["bot"]["hp"], entry["bot"]["attack"], entry["bot"]["heal"], int(entry["bot"]["block"])
        ]
        action = entry["bot"]["action"] - 1
        X.append(features)
        y.append(action)
    return np.array(X), np.array(y)

class BattleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BattleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)

model = BattleNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

criterion = nn.CrossEntropyLoss()

losses = []

for _ in tqdm(range(NUM_GAMES), desc="Evaluating games"):
    game_data = generate_game(num_rounds=random.randint(20, 100))
    X, y = prepare_data(game_data)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        losses.append(loss.item())

avg_loss = sum(losses) / len(losses)

with open(LOSSLOGSAVEPATH, "w") as file:
    file.write(str(avg_loss))

print(f"Average Loss over {NUM_GAMES} games: {avg_loss:.4f}")
