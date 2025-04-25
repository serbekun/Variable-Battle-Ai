# Variable Battle AI

Variable Battle AI is a neural network-powered bot designed to play the text-based game Variable Battle. It utilizes PyTorch for deep learning and logs gameplay data for further training and analysis.

## Features
- AI-controlled bot for Variable Battle
- Neural network implemented with PyTorch
- Logs game data for training and improvement
- Interactive gameplay where a player competes against the AI

## use import
- torch
- json
- os
- random
- sklearn.model_selection

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/serbekun/Variable-Battle-Ai.git
   cd Variable-Battle-Ai
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure the core game files (`cores.vbc`, `cores.vba`) are available.

## Usage

- for Run the script for start play the game with AI:
- in start in file ```/test_model_place/vbm.py```
have const with model name chnage this for change play model
- then star script
 ```sh
  /test_model_place/vbm.py
  ```
- use
  ```sh
  python3 vbm.py
  ```
 
During the game, choose actions using the corresponding keys:
- `1` or `a` - Attack
- `2` or `h` - Heal
- `3` or `b` - Block
- `4` or `ia` - Increase Attack
- `5` or `ih` - Increase Heal

## AI Model
The AI uses a fully connected neural network with the following parameters:
- **Input size**: 9
- **Hidden layers**: 5 layers of 126 neurons
- **Output size**: 5 (representing possible actions)

all exist AI models

vb_model1.pth       # first model can easy loss all
vb_model2.pth       # second model can so so play
vb_model3.pth       # thre model can play best vb_model2.pth
vb_model_data_1.pth # model create with date pack data_1.json

## Project Structure
```
Variable-Battle-Ai/
│
├── README.md                        # Project documentation
│
├── create_model/                   # Folder containing model training scripts
│   ├── create_model.py             # Main script for training models with settings
│   └── create_model_ujf.py         # Additional script for creating and training models
│
├── date_generator/                 # Generator for synthetic data used in training
│   └── date_generator.py           # Script to generate game data dynamically
│
├── date_packs/                     # Folder containing JSON data packs for training
│   └── data_1.json                 # Training data pack (version 1)
│
├── game_logs/                      # Folder containing logs of games played
│   ├── vb_model1.json              # Game log from model version 1
│   ├── vb_model2.json              # Test games played with model version 2
│   ├── vb_model3.json              # Game log from model version 3
│   ├── vb_model_data_1.json        # Additional data collected from games
│   ├── vb_model_example2.json      # Example model log (version 2)
│   └── vb_model_from_json.json     # Log generated from JSON-trained model
│
├── models/                         # Folder for storing trained model weights (.pth)
│   ├── vb_model1.pth               # Trained weights for model version 1
│   ├── vb_model2.pth               # Trained weights for model version 2
│   ├── vb_model3.pth               # Trained weights for model version 3
│   ├── vb_model_data_1.pth         # Model trained on data_1.json
│   ├── vb_model_date_2.pth         # Model trained on date pack version 2
│   ├── vb_model_example2.pth       # Example trained model version 2
│   ├── vb_model_example3.pth       # Example trained model version 3
│   ├── vb_model_example.pth        # General example model
│   └── vb_model_from_json.pth      # Model trained on data from JSON
│
├── models_loss_log/                # Training loss logs for each model
│   ├── vb_model1.json              # Loss log for model version 1
│   ├── vb_model2.json              # Loss log for model version 2
│   ├── vb_model3.json              # Loss log for model version 3
│   ├── vb_model_data_1.json        # Loss log for data_1 model
│   ├── vb_model_date_2.json        # Loss log for date_2 model
│   ├── vb_model_example2.json      # Loss log for example model 2
│   ├── vb_model_example3.json      # Loss log for example model 3
│   ├── vb_model_example.json       # Loss log for general example model
│   └── vb_model_from_json.json     # Loss log for JSON-trained model
│
├── test_model_loss/                # Testing script to evaluate model performance by loss
│   └── test_model_loss.py          # Script to compute and compare model losses
│
└── test_model_place/               # Folder for testing models in the game environment
    ├── cores/                      # Core game logic and utility functions
    │   ├── __pycache__/            # Cached bytecode for Python modules
    │   │   ├── vba.cpython-312.pyc
    │   │   ├── vba.cpython-38.pyc
    │   │   ├── vbc.cpython-312.pyc
    │   │   └── vbc.cpython-38.pyc
    │   ├── vba.py                  # Game logic and action definitions
    │   └── vbc.py                  # Core functionality (e.g. status display)
    └── vbm.py                      # Main script to run and test AI models in-game

```

## Function Overview
#### `log_game_data(...)`
Logs the current game state, including player and bot attributes, into a JSON file.

#### `predict_action(state)`
Uses the neural network to predict the bot's next action based on the game state.

#### `BattleNet(nn.Module)`
Defines the AI model architecture for decision-making.

#### `check_end_round(...)`
Determines if the game should continue or end.

#### `show_display(...)`
Displays the current state of the game.

#### Player Actions
- `player_attack_def(...)`
- `player_heal_def(...)`
- `player_block_def(...)`
- `player_increase_attack_def(...)`
- `player_increase_heal_def(...)`

#### Bot Actions
- `bot_attack_def(...)`
- `bot_heal_def(...)`
- `bot_block_def(...)`
- `bot_increase_attack_def(...)`
- `bot_increase_heal_def(...)`

## Contributing
Feel free to contribute by submitting issues, feature requests, or pull requests.

## License
This project is licensed under the MIT License.

## Author
Developed by [serbekun](https://github.com/serbekun).
