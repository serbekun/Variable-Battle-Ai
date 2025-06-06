import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys

import include.vbc as vbc 
import include.vba as vba  

PLAYER_MODELNAME = "vb_model_learn_infin"
BOT_MODELNAME = "vb_model_learn_infin"
GAME_LOG_SAVE = f"../logs/play_log/model_self_log/{PLAYER_MODELNAME}_{BOT_MODELNAME}.json"
PLAYER_MODEL_LOAD_PATH = f"../models/{PLAYER_MODELNAME}.pth"
BOT_MODEL_LOAD_PATH = f"../models/{BOT_MODELNAME}.pth"

LEARN_RATE = 0.1

# goto model file

parent_dir = os.path.abspath (os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# model
from model import Model as ModeL

def log_game_data(player_hp, player_attack, player_heal, player_block,
                  bot_hp, bot_attack, bot_heal, bot_block,
                  round_count, player_action, bot_action, filename=GAME_LOG_SAVE):
    log_entry = {
        "round_count": round_count,
        PLAYER_MODELNAME: {
            "hp": player_hp,
            "attack": player_attack,
            "heal": player_heal,
            "block": player_block,
            "action": player_action
        },
        BOT_MODELNAME: {
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
    return loss.item()

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

player_model = ModeL(input_size=9, hidden_size=126, output_size=5)
bot_model = ModeL(input_size=9, hidden_size=126, output_size=5)
criterion = nn.CrossEntropyLoss()
optimizer_player = optim.Adam(player_model.parameters(), lr=LEARN_RATE)
optimizer_bot = optim.Adam(bot_model.parameters(), lr=LEARN_RATE)

if os.path.exists(PLAYER_MODEL_LOAD_PATH):
    player_model.load_state_dict(torch.load(PLAYER_MODEL_LOAD_PATH))
    player_model.eval()
    print("Player model loaded.")
else:
    print("Player model not found. Will train from scratch.")

if os.path.exists(BOT_MODEL_LOAD_PATH):
    bot_model.load_state_dict(torch.load(BOT_MODEL_LOAD_PATH))
    bot_model.eval()
    print("Bot model loaded.")
else:
    print("Bot model not found. Will train from scratch.")

round_count = 0
player_hp, player_attack, player_heal, player_block = 100, 5, 5, False
bot_hp, bot_attack, bot_heal, bot_block = 100, 5, 5, False

best_loss_player = float('inf')
best_loss_bot = float('inf')

game_count = 0
NUM_GAMES = 1000

while game_count < NUM_GAMES:

    if game_count % 50 == 0:
        print("epoch -", game_count)

    round_count = 0
    player_hp, player_attack, player_heal, player_block = 100, 5, 5, False
    bot_hp, bot_attack, bot_heal, bot_block = 100, 5, 5, False
    game_count += 1

    while round_count < 100:
        if not vbc.check_end_round(round_count, player_hp, bot_hp):
            break

        state = {
            "round_count": round_count,
            "player": {"hp": player_hp, "attack": player_attack, "heal": player_heal, "block": player_block, "action": None},
            "bot": {"hp": bot_hp, "attack": bot_attack, "heal": bot_heal, "block": bot_block}
        }

        player_action = predict_action(player_model, state)
        state["player"]["action"] = player_action
        if player_action == 0:
            bot_hp, bot_block = vba.player_attack_def(player_attack, bot_hp, bot_block)
        elif player_action == 1:
            player_hp = vba.player_heal_def(player_heal, player_hp)
        elif player_action == 2:
            player_block = vba.player_block_def(player_block)
        elif player_action == 3:
            player_attack = vba.player_increase_attack_def(player_attack)
        elif player_action == 4:
            player_heal = vba.player_increase_heal_def(player_heal)

        bot_action = predict_action(bot_model, state)
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

        log_game_data(player_hp, player_attack, player_heal, player_block,
                      bot_hp, bot_attack, bot_heal, bot_block,
                      round_count, player_action, bot_action, filename=GAME_LOG_SAVE)

        current_loss_player = train_model_on_single_step(player_model, optimizer_player, criterion, state, player_action)
        current_loss_bot = train_model_on_single_step(bot_model, optimizer_bot, criterion, state, bot_action)

        if current_loss_player < best_loss_player:
            best_loss_player = current_loss_player
            torch.save(player_model.state_dict(), PLAYER_MODEL_LOAD_PATH)
            print(f"New best {PLAYER_MODELNAME} loss: {best_loss_player:.4f} - model saved.")

        if current_loss_bot < best_loss_bot:
            best_loss_bot = current_loss_bot
            torch.save(bot_model.state_dict(), BOT_MODEL_LOAD_PATH)
            print(f"New best {BOT_MODELNAME} loss: {best_loss_bot:.4f} - model saved.")

        player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal = \
            vbc.cat_limits(player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal)

        round_count += 1