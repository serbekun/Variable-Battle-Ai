import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import include.vbc as vbc
import include.vba as vba
import json
import os
import sys

MODELNAME = "vb_model3_con_learn_wpa"
GAME_LOG_SAVE = f"../logs/play_logs/with_player/{MODELNAME}.json"
MODEL_LOAD_PATH = f"../models/{MODELNAME}.pth"
LEARN_RATE = 0.1

# goto model file

parent_dir = os.path.abspath (os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# model
from model import Model as ModeL

def log_game_data(player_hp, player_attack, player_heal, player_block,
                  bot_hp, bot_attack, bot_heal, bot_block,
                  round_count, player_action, bot_action, filename=GAME_LOG_SAVE):
    # Create a log entry with game data
    log_entry = {
        "round_count": round_count,
        "human": {
            "hp": player_hp,
            "attack": player_attack,
            "heal": player_heal,
            "block": player_block,
            "action": player_action
        },
        MODELNAME: {
            "hp": bot_hp,
            "attack": bot_attack,
            "heal": bot_heal,
            "block": bot_block,
            "action": bot_action
        }
    }
    
    # Append the log entry to the file
    with open(filename, "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")  # Add a newline for each new log entry

def train_model_on_single_step(model, optimizer, criterion, state, action):
    features = np.array([
        state["round_count"],
        state["player"]["hp"], state["player"]["attack"], state["player"]["heal"], int(state["player"]["block"]),
        state["bot"]["hp"], state["bot"]["attack"], state["bot"]["heal"], int(state["bot"]["block"])
    ])
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    target = torch.tensor([action], dtype=torch.long)

    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

def predict_action(model, state):
    features = np.array([
        state["round_count"],
        state["player"]["hp"], state["player"]["attack"], state["player"]["heal"], int(state["player"]["block"]),
        state["bot"]["hp"], state["bot"]["attack"], state["bot"]["heal"], int(state["bot"]["block"])
    ])
    with torch.no_grad():
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        _, predicted_action = torch.max(output, 1)
    return predicted_action.item()

model = ModeL(input_size=9, hidden_size=126, output_size=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

if os.path.exists(MODEL_LOAD_PATH):
    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
    model.eval()
    print("Model loaded.")
else:
    print("Model not found. Will train from scratch.")

round_count = 0
player_hp, player_attack, player_heal, player_block = 100, 5, 5, False
bot_hp, bot_attack, bot_heal, bot_block = 100, 5, 5, False

count = 0

while count < 100:
    round_count = 0
    player_hp, player_attack, player_heal, player_block = 100, 5, 5, False
    bot_hp, bot_attack, bot_heal, bot_block = 100, 5, 5, False
    count += 1
    while round_count < 100:
        if not vbc.check_end_round(round_count, player_hp, bot_hp):
            torch.save(model.state_dict(), MODEL_LOAD_PATH)
            print("Game ended. Model saved.")
            break

        vbc.show_display(player_hp, player_attack, player_heal, player_block,
                        bot_hp, bot_attack, bot_heal, bot_block, round_count)

        player_action = input(":")
        if player_action in ("1", "a"):
            bot_hp, bot_block = vba.player_attack_def(player_attack, bot_hp, bot_block)
        elif player_action in ("2", "h"):
            player_hp = vba.player_heal_def(player_heal, player_hp)
        elif player_action in ("3", "b"):
            player_block = vba.player_block_def(player_block)
        elif player_action in ("4", "ia"):
            player_attack = vba.player_increase_attack_def(player_attack)
        elif player_action in ("5", "ih"):
            player_heal = vba.player_increase_heal_def(player_heal)
        else:
            print("Turn skipped!")

        state = {
            "round_count": round_count,
            "player": {"hp": player_hp, "attack": player_attack, "heal": player_heal, "block": player_block, "action": player_action},
            "bot": {"hp": bot_hp, "attack": bot_attack, "heal": bot_heal, "block": bot_block}
        }
        bot_action = predict_action(model, state)


        if bot_action == 0:
            player_hp, player_block = vba.bot_attack_def(bot_attack, player_hp, player_block)
        elif bot_action == 1:
            bot_hp = vba.bot_heal_def(bot_heal, bot_hp)
        elif bot_action == 2:
            bot_block = vba.bot_block_def(bot_block)
        elif bot_action == 3:
            bot_attack = vba.bot_increase_attack_def(bot_attack)
        elif bot_action == 4:
            bot_heal = vba.bot_increase_heal_def(bot_heal)

        player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal = \
            vbc.cat_limits(player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal)

        log_game_data(player_hp, player_attack, player_heal, player_block,
                    bot_hp, bot_attack, bot_heal, bot_block,
                    round_count, player_action, bot_action, filename=GAME_LOG_SAVE)

        # train
        print("train_model")
        train_model_on_single_step(model, optimizer, criterion, state, bot_action)

        round_count += 1
