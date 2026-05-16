# 📦 Board Game Package
[`Board-Game-AI\`](https://github.com/n-smith-byu/Board-Game-AI/)[`src\board_game\`](https://github.com/n-smith-byu/Board-Game-AI/tree/main/src/board_game)

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
