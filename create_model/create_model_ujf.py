import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
from sklearn.model_selection import train_test_split

# Neural network class
class BattleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
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

# Function to generate game data
def generate_game(num_rounds=100):
    game_data = []
    player_hp, bot_hp = 100, 100
    player_attack, player_heal = 5, 5
    bot_attack, bot_heal = 5, 5
    player_block, bot_block = False, False

    for round_count in range(num_rounds):
        player_action = random.randint(1, 5)
        bot_action = random.randint(1, 5)

        if player_action == 1:  # Attack
            if not bot_block:
                bot_hp -= player_attack
            bot_block = False
        elif player_action == 2:  # Heal
            player_hp += player_heal
        elif player_action == 3:  # Block
            player_block = True
        elif player_action == 4:  # Upgrade attack
            player_attack += 5
        elif player_action == 5:  # Upgrade heal
            player_heal += 5

        if bot_action == 1:  # Attack
            if not player_block:
                player_hp -= bot_attack
            player_block = False
        elif bot_action == 2:  # Heal
            bot_hp += bot_heal
        elif bot_action == 4:  # Upgrade attack
            bot_attack += 5
        elif bot_action == 5:  # Upgrade heal
            bot_heal += 5

        game_data.append([
            round_count,
            player_hp, player_attack, player_heal, int(player_block),
            bot_hp, bot_attack, bot_heal, int(bot_block),
            bot_action - 1
        ])
    return game_data

# Load or create the model
def load_or_create_model(input_size, hidden_size, output_size, model_name):
    model = BattleNet(input_size, hidden_size, output_size)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
    return model

# Dynamic progress update function
def print_status(epoch, total_epochs, loss, best_loss, gen, total_gen):
    sys.stdout.write("\033[F" * 3)  # Move up 3 lines
    sys.stdout.write(f"Epoch {epoch}/{total_epochs}\n")
    sys.stdout.write(f"Loss: {loss:.4f} | Best: {best_loss:.4f}\n")
    sys.stdout.write(f"Generating {gen}/{total_gen}\n")
    sys.stdout.flush()

# Batch processing
def yield_batch(model, X, y):
    global best_loss
    model.train()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.1)

    total_loss = 0.0
    for i in range(0, len(X_train), BATCH_SIZE):
        xb = X_train[i:i + BATCH_SIZE]
        yb = y_train[i:i + BATCH_SIZE]

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(X_train) // BATCH_SIZE)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODELSAVEPATH)

    return avg_loss, val_loss.item()

# Training function
def train_model(model):
    global best_loss
    print("\n\n\n")  # Space for 3 dynamically updated lines
    for epoch in range(EPOCHS):
        X_batches, y_batches = [], []

        for gen_index in range(GAMES_PER_EPOCH):
            data = generate_game(random.randint(20, 100))
            for entry in data:
                X_batches.append(entry[:9])
                y_batches.append(entry[9])
                if len(X_batches) >= BATCH_SIZE:
                    avg_loss, val_loss = yield_batch(model, X_batches, y_batches)
                    print_status(epoch + 1, EPOCHS, avg_loss, best_loss, gen_index + 1, GAMES_PER_EPOCH)
                    X_batches.clear()
                    y_batches.clear()

        if X_batches:
            avg_loss, val_loss = yield_batch(model, X_batches, y_batches)
            print_status(epoch + 1, EPOCHS, avg_loss, best_loss, GAMES_PER_EPOCH, GAMES_PER_EPOCH)

# Training parameters
INPUT_SIZE = 9
HIDDEN_SIZE = 126
OUTPUT_SIZE = 5
EPOCHS = 5
GAMES_PER_EPOCH = 100
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
MODEL_NAME = "vb_model_example.pth"
MODELSAVEPATH = "../models/" + MODEL_NAME

# Initialize model
model = load_or_create_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MODEL_NAME)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_loss = float('inf')

# Start training
train_model(model)
