import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
from sklearn.model_selection import train_test_split

INPUT_SIZE = 9
HIDDEN_SIZE = 126
OUTPUT_SIZE = 5
EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
MODEL_NAME = "vb_model_example.pth"
MODEL_SAVE_PATH = "../models/" + MODEL_NAME
JSON_PATH = "../date_packs/data_example.json"

# ==== Модель ====
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

def load_or_create_model():
    model = BattleNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("model loaded from fail.")
    else:
        print("create new model")
    return model

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        sessions = json.load(f)

    X, y = [], []
    for session in sessions:
        for round_data in session:
            human = round_data["human"]
            bot = round_data["vb_model1"]

            input_vector = [
                round_data["round_count"],
                human["hp"], human["attack"], human["heal"], int(human["block"]),
                bot["hp"], bot["attack"], bot["heal"], int(bot["block"])
            ]
            action = bot["action"] - 1

            X.append(input_vector)
            y.append(action)

    return X, y

def print_status(epoch, total_epochs, loss, val_loss, best_loss):
    sys.stdout.write("\033[F" * 3)
    sys.stdout.write(f"Epoch {epoch}/{total_epochs}\n")
    sys.stdout.write(f"Loss: {loss:.4f} | Val: {val_loss:.4f} | Best: {best_loss:.4f}\n")
    sys.stdout.write("\n")
    sys.stdout.flush()

def train_model(model, X, y):
    global best_loss

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.1)

        total_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            xb = X_train[i:i+BATCH_SIZE]
            yb = y_train[i:i+BATCH_SIZE]

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
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print_status(epoch, EPOCHS, avg_loss, val_loss.item(), best_loss)

if __name__ == "__main__":
    print("load date from file", JSON_PATH)
    X_data, y_data = load_data_from_json(JSON_PATH)

    model = load_or_create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf')

    print("\n\n\n")
    train_model(model, X_data, y_data)
