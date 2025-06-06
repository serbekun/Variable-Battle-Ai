import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
from tqdm import tqdm

INPUT_SIZE = 9
HIDDEN_SIZE = 126
OUTPUT_SIZE = 5
EPOCHS = 30
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
MODEL_NAME = "vb_model_dg2_smart_clat.pth"
MODEL_NAME_IN_JSON = "model"
MODEL_SAVE_PATH = os.path.join("..", "models", MODEL_NAME)
NDJSON_PATH = os.path.join("..", "date_packs", "data_dg2_1.ndjson")

# init cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use device: {device}")

# goto model file

parent_dir = os.path.abspath (os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# model
from model import Model as ModeL

def load_or_create_model():
    model = ModeL(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("Model loaded from file.")
    else:
        print("Creating new model.")
    return model

def stream_ndjson(file_path, batch_size):

    X, y = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                round_data = json.loads(line)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                continue
            human = round_data["human"]
            bot = round_data[MODEL_NAME_IN_JSON]
            input_vector = [
                round_data["round_count"],
                human["hp"], human["attack"], human["heal"], int(human["block"]),
                bot["hp"], bot["attack"], bot["heal"], int(bot["block"])
            ]
            action = bot["action"]
            if action <= 0:
                continue

            X.append(input_vector)
            y.append(action)

            if len(X) >= batch_size:
                yield X, y
                X, y = [], []
        if X:
            yield X, y

def print_status(epoch, total_epochs, train_loss, val_loss, best_loss):
    sys.stdout.write("\033[F" * 3)
    sys.stdout.write(f"Epoch: {epoch}/{total_epochs}\n")
    sys.stdout.write(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_loss:.4f}\n")
    sys.stdout.write("\n")
    sys.stdout.flush()

def evaluate_model(model, file_path, batch_size, criterion, max_batches=10):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for X_batch, y_batch in stream_ndjson(file_path, batch_size):
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            total_loss += loss.item()
            count += 1
            if count >= max_batches:
                break
    return total_loss / count if count > 0 else 0.0

def train_model(model, file_path):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for X_batch, y_batch in stream_ndjson(file_path, BATCH_SIZE):
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        train_loss = running_loss / batch_count if batch_count > 0 else 0.0
        val_loss = evaluate_model(model, file_path, BATCH_SIZE, criterion)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print_status(epoch, EPOCHS, train_loss, val_loss, best_loss)

if __name__ == "__main__":
    print(f"Training data will be loaded in a streaming mode from: {NDJSON_PATH}")
    model = load_or_create_model()
    print("\nStarting Training...\n")
    train_model(model, NDJSON_PATH)