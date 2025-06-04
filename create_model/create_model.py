import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal):
    if bot_hp < 50:
        return 2
    elif player_hp < 40:
        return 1
    elif round_count > 60:
        return 1
    elif round_count < 100:
        inc_heal_or_attack = random.randint(1, 2)
        if inc_heal_or_attack == 1 and bot_attack < 50:
            return 4
        elif inc_heal_or_attack == 2 and bot_hp < 50:
            return 5
        else:
            return inc_heal_or_attack
    else:
        return random.randint(1, 5)

def generate_game(num_rounds=100):
    game_data = []
    player_hp, bot_hp = 100, 100
    player_attack, player_heal = 5, 5
    bot_attack, bot_heal = 5, 5
    player_block, bot_block = False, False

    for round_count in range(num_rounds):
        player_action = get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal)
        bot_action = get_action(player_hp, bot_hp, round_count, bot_attack, bot_heal)

        if player_action == 1:
            if not bot_block:
                bot_hp -= player_attack
            bot_block = False
        elif player_action == 2:
            player_hp += player_heal
        elif player_action == 3:
            player_block = True
        elif player_action == 4:
            player_attack += 5
        elif player_action == 5:
            player_heal += 5

        if bot_action == 1:
            if not player_block:
                player_hp -= bot_attack
            player_block = False
        elif bot_action == 2:
            bot_hp += bot_heal
        elif bot_action == 4:
            bot_attack += 5
        elif bot_action == 5:
            bot_heal += 5

        game_data.append([
            round_count,
            player_hp, player_attack, player_heal, int(player_block),
            bot_hp, bot_attack, bot_heal, int(bot_block),
            bot_action - 1
        ])

    return game_data


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
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


def load_or_create_model(input_size, hidden_size, output_size, model_name):
    model = BattleNet(input_size, hidden_size, output_size)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
    return model


def print_status(epoch, total_epochs, loss, val_loss, best_loss, gen, total_gen):
    sys.stdout.write("\033[F" * 3)
    sys.stdout.write(f"Epoch {epoch}/{total_epochs}\n")
    sys.stdout.write(f"Loss: {loss:.4f} | Best: {best_loss:.4f}\n")
    sys.stdout.write(f"Generating {gen}/{total_gen}\n")
    sys.stdout.flush()


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


def train_model(model):
    global best_loss

    print("\n\n\n")
    for epoch in range(EPOCHS):
        X_batches, y_batches = [], []

        for gen_index in range(GAMES_PER_EPOCH):
            data = generate_game(random.randint(20, 100))
            for entry in data:
                X_batches.append(entry[:9])
                y_batches.append(entry[9])
                if len(X_batches) >= BATCH_SIZE:
                    avg_loss, val_loss = yield_batch(model, X_batches, y_batches)
                    print_status(epoch + 1, EPOCHS, avg_loss, val_loss, best_loss, gen_index + 1, GAMES_PER_EPOCH)
                    X_batches.clear()
                    y_batches.clear()

        if X_batches:
            avg_loss, val_loss = yield_batch(model, X_batches, y_batches)
            print_status(epoch + 1, EPOCHS, avg_loss, val_loss, best_loss, GAMES_PER_EPOCH, GAMES_PER_EPOCH)


INPUT_SIZE = 9
HIDDEN_SIZE = 126
OUTPUT_SIZE = 5
EPOCHS = 500_000
GAMES_PER_EPOCH = 100
BATCH_SIZE = 1024
LEARNING_RATE = 0.01
MODEL_NAME = "vb_model_kenta1.pth"
MODELSAVEPATH = "../models/" + MODEL_NAME

model = load_or_create_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, MODEL_NAME)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_loss = float('inf')

train_model(model)
