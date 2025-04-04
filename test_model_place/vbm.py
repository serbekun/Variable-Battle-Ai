import torch
import torch.nn as nn
import numpy as np
import joblib
import cores.vbc as vbc
import cores.vba as vba
import json

GAMELOGSAVE = "game_log_vb_model2.json"
MODELNAME = "vb_model3.pth"

def log_game_data(player_hp, player_attack, player_heal, player_block,
                  bot_hp, bot_attack, bot_heal, bot_block,
                  round_count, player_action, bot_action, filename=GAMELOGSAVE):
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

# Define the neural network model
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

# Load the model
model = BattleNet(input_size=9, hidden_size=126, output_size=5)

# Load saved model weights
model.load_state_dict(torch.load('cores/vb_model2.pth'))
model.eval()  # Set the model to evaluation mode

# Function to predict bot's action
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

# Initialize game state variables
round_count = 0

# Player attributes
player_hp = 100
player_attack = 5
player_heal = 5
player_block = False
player_action = 0

# Bot attributes
bot_hp = 100
bot_attack = 5
bot_heal = 5
bot_block = False
bot_action = 0

while True:
    # Check if the game should continue
    if vbc.check_end_round(round_count, player_hp, bot_hp) == False:
        print("Exiting")
        exit(1)
        
    vbc.show_display(player_hp, player_attack, player_heal, player_block, bot_hp, bot_attack, bot_heal, bot_block, round_count)
    
    # Player selects an action
    player_action = input(":")

    if player_action == "1" or player_action == "a":
        player_now_attack = True
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

    # Update game state
    state = {
        "round_count": round_count,
        "player": {
            "hp": player_hp,
            "attack": player_attack,
            "heal": player_heal,
            "block": player_block,
            "action": player_action
        },
        "bot": {
            "hp": bot_hp,
            "attack": bot_attack,
            "heal": bot_heal,
            "block": bot_block,
            "action": bot_action
        }
    }

    # Predict bot's action
    bot_action = predict_action(state)

    # Perform bot's action
    if bot_action == 0:
        player_hp, player_block = vba.bot_attack_def(bot_attack, player_hp, player_block)
    if bot_action == 1:
        bot_hp = vba.bot_heal_def(bot_heal, bot_hp)
    if bot_action == 2:
        bot_block = vba.bot_block_def(bot_block)
    if bot_action == 3:
        bot_attack = vba.bot_increase_attack_def(bot_attack)
    if bot_action == 4:
        bot_heal = vba.bot_increase_heal_def(bot_heal)
        
    player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal = vbc.cat_limits(player_hp, player_attack, player_heal, bot_hp, bot_attack, bot_heal)

    # Log game data
    log_game_data(player_hp, player_attack, player_heal, player_block,
                  bot_hp, bot_attack, bot_heal, bot_block,
                  round_count, player_action, bot_action, filename=GAMELOGSAVE)
    
    round_count += 1
