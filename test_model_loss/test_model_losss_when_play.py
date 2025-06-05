# not work

import torch
import numpy as np
import torch.nn as nn
import os
import include.vbc as vbc
import include.vba as vba

MODEL_LOAD_PATH = "../models/vb_model3_con_learn_wpa.pth" # ---------CHECK--------------------

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

def predict_action(state):
    # Convert state into a format suitable for the model
    features = np.array([state["round_count"],
                         state["player"]["hp"], state["player"]["attack"], state["player"]["heal"], int(state["player"]["block"]),
                         state["bot"]["hp"], state["bot"]["attack"], state["bot"]["heal"], int(state["bot"]["block"])])
    
    # Make prediction using the model
    with torch.no_grad():
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
        _, predicted_action = torch.max(output, 1)
    
    return predicted_action.item()  # Return predicted action

# init model
model = BattleNet(input_size=9, hidden_size=126, output_size=5)
criterion = nn.CrossEntropyLoss()

if os.path.exists(MODEL_LOAD_PATH):
    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
    model.eval()
    print(f"Model {MODEL_LOAD_PATH} loaded.")
else:
    print("Model not loaded, using a new model.")

round_count = 0
player_hp, player_attack, player_heal, player_block = 100, 5, 5, False
bot_hp, bot_attack, bot_heal, bot_block = 100, 5, 5, False

loss_values = []

def predict_bot_action(state):
    return 0 if state["bot"]["hp"] > 50 else 1

while round_count < 100:
    if not vbc.check_end_round(round_count, player_hp, bot_hp):
        print("Game end.")
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
        "player": {"hp": player_hp, "attack": player_attack, "heal": player_heal, "block": player_block},
        "bot": {"hp": bot_hp, "attack": bot_attack, "heal": bot_heal, "block": bot_block}
    }

    bot_action = predict_action(state)
    bot_action = torch.tensor([bot_action], dtype=torch.long)

    with torch.no_grad():
        input_tensor = torch.tensor([
            state["round_count"],
            state["player"]["hp"],
            state["player"]["attack"],
            state["player"]["heal"],
            int(state["player"]["block"]),
            state["bot"]["hp"],
            state["bot"]["attack"],
            state["bot"]["heal"],
            int(state["bot"]["block"])
        ], dtype=torch.float32).unsqueeze(0)

        outputs = model(input_tensor)
        loss = criterion(outputs, bot_action)
        loss_values.append(loss.item())

    print(f"Loss: {loss.item()}")
    round_count += 1

if loss_values:
    average_loss = sum(loss_values) / len(loss_values)
    print(f"\nstable loss: {average_loss:.4f}")