# Variable Battle AI

Variable Battle AI is a neural network-powered bot designed to play the text-based game Variable Battle. It utilizes PyTorch for deep learning and logs gameplay data for further training and analysis.

## Features

* AI-controlled bot for Variable Battle
* Neural network implemented with PyTorch
* Logs game data for training and improvement
* Interactive gameplay where a player competes against the AI

## Imports

* torch
* json
* os
* random
* sklearn.model\_selection

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/serbekun/Variable-Battle-Ai.git
   cd Variable-Battle-Ai
   ```
2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```
3. Ensure the core game files (`cores.vbc`, `cores.vba`) are available.

## Usage

* To start playing the game with the AI:
* In the file `test_model_place/vbm.py`, change the constant that specifies the model name to select the desired model.
* Then, run the script:

  ```sh
  python3 test_model_place/vbm.py
  ```

During the game, choose actions using the following keys:

* `1` or `a` — Attack
* `2` or `h` — Heal
* `3` or `b` — Block
* `4` or `ia` — Increase Attack
* `5` or `ih` — Increase Heal

## AI Model

The AI uses a fully connected neural network with the following parameters:

* **Input size**: 9
* **Hidden layers**: 5 layers with 126 neurons each
* **Output size**: 5 (representing possible actions)

Available AI models:

* `vb_model1.pth`: Basic model, often loses.
* `vb_model2.pth`: Intermediate model, performs moderately well.
* `vb_model3.pth`: More advanced, plays better than `vb_model2.pth`.
* `vb_model_data_1.pth`: Model trained using `data_1.json` dataset.

## Project Structure

```
Variable-Battle-Ai
│  README.md                        # Project documentation
│  model.py                         # AI model architecture
│
├── create_model/                   # Scripts for creating and training models
│   ├── create_model.py
│   ├── create_model_ujf.py
│   ├── create_model_when_play.py
│   ├── create_model_with_another_model.py
│
├── date_generator/                 # Data generation scripts for training
│   ├── date_generator2.py
│   ├── date_generator_action_from_player.py
│
├── date_packs/                     # Datasets for training
│   ├── data_1_from_player.ndjson
│   ├── data_fp_attack_train.ndjson
│
├── game_logs/                      # Logs of gameplay and training
│   ├── vb_model1.json
│   ├── vb_model2.json
│   ├── vb_model3.json
│   ├── vb_model_data_1.json
│   ├── vb_model_example2.json
│   ├── vb_model_example3.json
│   ├── vb_model_from_json.json
│
├── models/                         # Saved AI models (.pth files)
│   ├── vb_model1.pth
│   ├── vb_model2.pth
│   ├── vb_model3.pth
│   ├── vb_model_data_1.pth
│   ├── vb_model_date_2.pth
│   ├── vb_model_example.pth
│   ├── vb_model_example2.pth
│   ├── vb_model_example3.pth
│   ├── vb_model_from_json.pth
│
├── models_loss_log/                # Training loss logs
│   ├── vb_model1.json
│   ├── vb_model2.json
│   ├── vb_model3.json
│   ├── vb_model_data_1.json
│   ├── vb_model_date_2.json
│   ├── vb_model_example.json
│   ├── vb_model_example2.json
│   ├── vb_model_example3.json
│   ├── vb_model_from_json.json
│
├── test_model_loss/                # Scripts for evaluating model loss
│   ├── test_model_loss.py
│
└── test_model_place/               # Game testing with AI
    ├── cores/
    │   ├── vba.py
    │   ├── vbc.py
    ├── vbm.py
```

## C++ Programs

To compile C++ programs:

```bash
g++ -std=c++17 -O3 -pthread -o date_generator2 date_generator2.cpp
```

## Function Overview

### `log_game_data(...)`

Logs the current game state, including player and bot attributes, to a JSON file.

### `predict_action(state)`

Uses the neural network to predict the bot's next action based on the current game state.

### `BattleNet(nn.Module)`

Defines the AI model architecture for decision-making.

### `check_end_round(...)`

Determines if the game should continue or end.

### `show_display(...)`

Displays the current game state.

### Player Actions

* `player_attack_def(...)`
* `player_heal_def(...)`
* `player_block_def(...)`
* `player_increase_attack_def(...)`
* `player_increase_heal_def(...)`

### Bot Actions

* `bot_attack_def(...)`
* `bot_heal_def(...)`
* `bot_block_def(...)`
* `bot_increase_attack_def(...)`
* `bot_increase_heal_def(...)`

## Contributing

Feel free to contribute by submitting issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License.

## Author

Developed by [serbekun](https://github.com/serbekun).
