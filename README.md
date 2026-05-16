# Board Game AI 
![Status: Active Development](https://img.shields.io/badge/Status-Active_Development-green) \
[![Python Version](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/)

Building AI Agents to play various board games using Reinforcement Learning.

## 🛠️ Project Status
This project is **actively being developed**. I am currently focusing on implementing the core logic and RL environments for the games listed below.

## 📦 Board Game Package
[`src\board_game\`](https://github.com/n-smith-byu/Board-Game-AI/tree/main/src/board_game)

A base package with classes for making Board Game implementation easier.
- `BoardGame`: The base Board Game class. Holds logic for adding and removing players and starting the game. 
- `GameBoard`: The base Game Board class. Minimal implementation. Abstract methods/API for retrieving a list of valid moves for a player.
- `WaitingRoom`: Convenient Tool for setting up a new game without knowing the player list ahead of time.
   - Compatible with any subclass of `BoardGame`.
   - Uses default assigned classes for bots and human players. 
   - Example Usage:
     ```python
     from board_game import WaitingRoom
     
     wr = WaitingRoom(<BoardGame:SubClass>)

     wr.add_bot()                  # adds a bot
     wr.add_player('Player2')      # add a human player with username Player2
     wr.add_bot()
   
     game = wr.create_game()       # initialize the game with the current player list
     game.run()
     ```
     
## 🎲 Games in Progress

### 1. Photosynthesis
[`src\photosynthesis\`](https://github.com/n-smith-byu/Board-Game-AI/tree/main/src/photosynthesis)

Based on the Photosynthesis game by [Blue Orange Games](https://www.blueorangegames.com/games/photosynthesis).
* **Status:** Implementing Deep Q-Learning w/ Monte Carlo Tree Search
* **Tech Stack:** Python

* **Alternate Method:**
    * Here is a write-up for an alternate method where I show how to reduce the rules to a set of Linear Constraints and create a Linear Programming Problem.
    * Link:
      * [`CS412_Final_Project_Photosynthesis_Linear_Programming.pdf`](https://n-smith-byu.github.io/Board-Game-AI/src/photosynthesis/docs/CS412_Final_Project_Photosynthesis_Linear_Programming.pdf)

---

## ⚖️ Licensing

This repository uses a split-licensing model:
* **Framework:** The core engine and base classes in `src/board_game` are licensed under the MIT License.
* **Game Implementations:** The code in `src/photosynthesis` is provided under a custom Source-Available license for personal, educational, and research purposes only. It does not carry a permissive open-source license to respect the intellectual property and trademarks of the original game creators.

