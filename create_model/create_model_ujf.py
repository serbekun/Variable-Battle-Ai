import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from sklearn.model_selection import train_test_split

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

def load_data_from_json(file_name):
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    
    X, y = [], []
    for session in data:
        for entry in session:
            features = [
                entry["round_count"],
                entry["human"]["hp"], entry["human"]["attack"], entry["human"]["heal"], int(entry["human"]["block"]),
                entry["vb_model1"]["hp"], entry["vb_model1"]["attack"], entry["vb_model1"]["heal"], int(entry["vb_model1"]["block"])
            ]
            action = entry["vb_model1"]["action"] - 1
            X.append(features)
            y.append(action)
    
    return np.array(X), np.array(y)

def train_model(model, epochs, batch_size, learning_rate, best_model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"epoch {epoch + 1}/{epochs}")

        X, y = load_data_from_json(LOAD_FILE_PATH)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.1)

        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(X_train) // batch_size)
        print(f"normal loss: {avg_loss:.4f}")

        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            print(f"loss : {val_loss.item():.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_name)
            print(f"new best model save: '{best_model_name}' with loss - {best_loss:.4f}")

# Настройки модели
INPUT_SIZE = 9
HIDDEN_SIZE = 126
OUTPUT_SIZE = 5
EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
BEST_MODEL_NAME = "vb_model_data_1.pth"
LOAD_FILE_NAME = "data_1.json"
LOAD_FILE_PATH = "../date_packs/" + LOAD_FILE_NAME
MODEL_SAVE_PATH = "../models/" + BEST_MODEL_NAME

model = BattleNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
train_model(model, EPOCHS, BATCH_SIZE, LEARNING_RATE, BEST_MODEL_NAME)
