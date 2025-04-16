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

## Project Structure
```
Variable-Battle-Ai/
|
|── logs                  # place for save all game logs
|   |── vb_model1.json    #
|   |── vb_model2.json    # test game with ai game logs
|   |── vb_model3.json    # 
|
│── create_model          # folder where exist cod for lerning model
│   │── create_model.py   # lerning model script with setting 
│
│── models/               # AI models and training data
│   │── vb_model1.pth     # Trained model weights firts vesrion
|   |── vb_model2.pth     # Trained model weights second version
|   |── vb_model3.pth     # Trained model weights thid version
│
|── test_model_place/     # AI test script folder
|   |── cores/
|   |        |── vba.py   # game logick with all action function
|   |        |── vbc.py   # game core with function for show status and cat limits
|   |── vbm.py            # main fail what need to start for play with model
|
│── README.md             # Project documentation
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
