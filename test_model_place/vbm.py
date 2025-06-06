import torch
import torch.nn as nn
import numpy as np
import joblib
import include.vbc as vbc
import include.vba as vba
import json
import sys
import os

# goto model file

parent_dir = os.path.abspath (os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# model
from model import Model as ModeL

MODELNAME = "vb_model_dg2_smart_clat"
GAME_LOG_SAVE = "../logs/play_log/with_player/" + MODELNAME + ".json"
MODEL_LOAD_PATH = "../models/" + MODELNAME + ".pth"

# init cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use device '{device}'")

def log_game_data(player_hp, player_attack, player_heal, player_block,
                  bot_hp, bot_attack, bot_heal, bot_block,
                  round_count, player_action, bot_action, filename=GAME_LOG_SAVE):
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

    with open(filename, "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")

model = ModeL(input_size=9, hidden_size=126, output_size=5).to(device)
model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
model.eval()

def predict_action(state):
    features = np.array([state["round_count"],
                         state["player"]["hp"], state["player"]["attack"], state["player"]["heal"], int(state["player"]["block"]),
                         state["bot"]["hp"], state["bot"]["attack"], state["bot"]["heal"], int(state["bot"]["block"])])

    with torch.no_grad():
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        _, predicted_action = torch.max(output, 1)

    return predicted_action.item()

round_count = 0

player_hp = 100
player_attack = 5
player_heal = 5
player_block = False
player_action = 0

bot_hp = 100
bot_attack = 5
bot_heal = 5
bot_block = False
bot_action = 0

while True:
    if vbc.check_end_round(round_count, player_hp, bot_hp) == False:
        print("Exiting")
        exit(1)

    vbc.show_display(player_hp, player_attack, player_heal, player_block, bot_hp, bot_attack, bot_heal, bot_block, round_count)

    player_action = input(":")

    if player_action == "1" or player_action == "a":
        bot_hp, bot_block = vba.player_attack_def(player_attack, bot_hp, bot_block)
    elif player_action == "2" or player_action == "h":
        player_hp = vba.player_heal_def(player_heal, player_hp)
    elif player_action == "3" or player_action == "b":
        player_block = vba.player_block_def(player_block)
    elif player_action == "4" or player_action == "ia":
        player_attack = vba.player_increase_attack_def(player_attack)
    elif player_action == "5" or player_action == "ih":
        player_heal = vba.player_increase_heal_def(player_heal)
    else:
        print("Turn skipped!")

    state = {
        "round_count": round_count,
        "player": {"hp": player_hp, "attack": player_attack, "heal": player_heal, "block": player_block, "action": player_action},
        "bot": {"hp": bot_hp, "attack": bot_attack, "heal": bot_heal, "block": bot_block, "action": bot_action}
    }

    bot_action = predict_action(state)

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

    player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal = vbc.cat_limits(player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal)

    log_game_data(player_hp, player_attack, player_heal, player_block,
                  bot_hp, bot_attack, bot_heal, bot_block,
                  round_count, player_action, bot_action, filename=GAME_LOG_SAVE)

    round_count += 1