import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os

# function for gene rate bot action
def get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal):
        
    if bot_hp < 50:
        return 2
    
    elif player_hp < 40:
        return 1
    
    elif round_count > 60:
        return 1

    elif round_count < 100 :
        inc_heal_or_attack = random.randint(1, 2)
        if inc_heal_or_attack == 1 and bot_attack < 50:
            return 4
        elif inc_heal_or_attack == 2 and bot_heal < 50:
            return 5
        
        else:
            return inc_heal_or_attack        
    
    else:
        return random.randint(1, 5)
    
# generation one game
def generate_game(num_rounds=100):
    game_data = []
    player_hp, bot_hp = 100, 100
    player_attack, player_heal = 5, 5
    bot_attack, bot_heal = 5, 5
    player_block, bot_block = False, False

    for round_count in range(num_rounds):
        player_action = get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal)
        bot_action = get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal)

        # processing actions
        if player_action == 1:
            if bot_block == True:
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
            if player_block == True:
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

# converting data to vector
def prepare_data(data):
    X, y = [], []
    for entry in data:
        features = [
            entry["round_count"],
            entry["player"]["hp"], entry["player"]["attack"], entry["player"]["heal"], int(entry["player"]["block"]),
            entry["bot"]["hp"], entry["bot"]["attack"], entry["bot"]["heal"], int(entry["bot"]["block"])
        ]
        action = entry["bot"]["action"] - 1  # select action from [1/5] to [0/4]
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)
    return X, y


# initialize values
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


# function for load or create ne model
def load_or_create_model(input_size, hidden_size, output_size, model_name):
    model = BattleNet(input_size, hidden_size, output_size)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
        print(f"Model '{model_name}' successful load")
    else:
        print(f"Model not found, create new model '{model_name}'")
    return model


# model setting
INPUT_SIZE = 9
HIDDEN_SIZE = 126
OUTPUT_SIZE = 5
EPOCHS = 100
GAMES_PER_EPOCH = 100000
BEST_MODEL_NAME = "vb_model3.pth"
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

# load or create model
model = load_or_create_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BEST_MODEL_NAME)

# loss and optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# initialize variable for print model loss
best_loss = float('inf')

# training model
def train_model(model, epochs=EPOCHS, games_per_epoch=GAMES_PER_EPOCH):
    global best_loss
    for epoch in range(epochs):
        print(f"🚀 epoch {epoch + 1}/{epochs} start")
        game_data = []

        # generation data
        for _ in range(games_per_epoch):
            game_data.extend(generate_game(num_rounds=random.randint(0, 100)))

        X, y = prepare_data(game_data)

        # converting data into tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # division of data into training and validation 
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.1)

        # training model batch
        epoch_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch = X_train[i:i + BATCH_SIZE]
            y_batch = y_train[i:i + BATCH_SIZE]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(X_train) // BATCH_SIZE)
        print(f"epoch {epoch + 1}/{epochs} end. average loss: {avg_loss:.4f}")

        # evaluation based on validation data
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            print(f"validation loss: {val_loss.item():.4f}")

        # save now model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_NAME)
            print(f"new best model save'{BEST_MODEL_NAME}' with loss: {best_loss:.4f}")

# call training model
train_model(model, epochs=EPOCHS, games_per_epoch=GAMES_PER_EPOCH)
