# Photosynthesis Board Game

[`Board-Game-AI\`](https://github.com/n-smith-byu/Board-Game-AI/)[`src\photosynthesis\`](https://github.com/n-smith-byu/Board-Game-AI/tree/main/src/photosynthesis)

![Status: Active Development](https://img.shields.io/badge/Status-Active_Development-green) \
[![Python Version](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/)

---

Implementation of an AI to play Photosynthesis by [Blue Orange Games](https://www.blueorangegames.com/games/photosynthesis).

> **Disclaimer:** 
> This project is an independent research endeavor focused on Artificial Intelligence and is not affiliated with Blue Orange Games. The game 'Photosynthesis' is a trademark of Blue Orange Games. This implementation is intended for educational and research purposes only.

### Quick Start

A sample game with 3 random players. Outputs the final score.



```python
from photosynthesis import PhotosynthesisGame, RandomPlayer

game = PhotosynthesisGame(players=[RandomPlayer(i) for i in range(3)], extra_round=False)
game.reset()
game.run()

```

### Human Player

The HumanPlayer is controlled through the terminal.

```python
from photosynthesis import HumanPlayer

player = HumanPlayer('<PlayerName>')
```

### AI Players

There are three available types of AI player in the works. 

1. RandomPlayer
    - Chooses random moves each turn. 
    - ```python
      from photosynthesis import RandomPlayer
  
      player = RandomPlayer(int: <player_num>)
      ```
3. RLPlayer (Coming...)
    - An AI player built using Deep Q-Learning with Monte Carlo Tree Search
4. LPlayer (Coming...)
    - An alternative approach using a method I developed in my Linear Programming Class, where I show how to reduce the rules to a set of Linear Constraints and transform a turn into a Linear Programming Problem. 
    - See `src\photosynthesis\docs\CS412_Final_Project_Photosynthesis_Linear_Programming_WriteUp.pdf`

### Playing the Game

If you are using the built-in HumanPlayer, then you will see a setup like this on your turn:

> **Note:** Make sure you have used `game.run(display=True)`, or you will not see the two boards each turn. 

```
Player3's Turn
Tree Sizes:
[[  1.  -1.   0.   1. -inf -inf -inf]
 [ -1.   2.  -1.  -1.  -1. -inf -inf]
 [  1.   0.  -1.  -1.  -1.  -1. -inf]
 [ -1.  -1.  -1.  -1.  -1.  -1.  -1.]
 [-inf  -1.   1.   0.  -1.  -1.  -1.]
 [-inf -inf   2.   1.   0.  -1.  -1.]
 [-inf -inf -inf   3.  -1.   1.   0.]] 

Player Occupancies
[[  1.  -1.   1.   1. -inf -inf -inf]
 [ -1.   1.  -1.  -1.  -1. -inf -inf]
 [  2.   1.  -1.  -1.  -1.  -1. -inf]
 [ -1.  -1.  -1.  -1.  -1.  -1.  -1.]
 [-inf  -1.   0.   2.  -1.  -1.  -1.]
 [-inf -inf   0.   2.   0.  -1.  -1.]
 [-inf -inf -inf   2.  -1.   0.   0.]] 

Please Choose Move From Available Options: 
Possible moves:
[0]: {'(pass_turn)'}
[1]: {"('buy_tree', size=0)"}
[2]: {"('buy_tree', size=1)"}
[3]: {"('buy_tree', size=2)"}
[4]: {"('plant_seed', pos=((1, 0), parent=(1, (2, 0))))"}
[5]: {"('plant_seed', pos=((3, 0), parent=(1, (2, 0))))"}
[6]: {"('plant_seed', pos=((3, 1), parent=(1, (2, 0))))"}
[7]: {"('plant_seed', pos=((6, 4), parent=(1, (5, 3))))"}
[8]: {"('plant_seed', pos=((3, 0), parent=(3, (6, 3))))"}
[9]: {"('plant_seed', pos=((3, 1), parent=(3, (6, 3))))"}
[10]: {"('plant_seed', pos=((3, 2), parent=(3, (6, 3))))"}
[11]: {"('plant_seed', pos=((3, 3), parent=(3, (6, 3))))"}
[12]: {"('plant_seed', pos=((4, 1), parent=(3, (6, 3))))"}
[13]: {"('plant_seed', pos=((4, 4), parent=(3, (6, 3))))"}
[14]: {"('plant_seed', pos=((5, 5), parent=(3, (6, 3))))"}
[15]: {"('plant_seed', pos=((6, 4), parent=(3, (6, 3))))"}
[16]: {"('grow_tree', size=0, pos=(4, 3))"}
[17]: {"('harvest_tree', pos=(6, 3))"}
Num Suns: 4
Input Choice:

```

#### Tree Sizes Board:

```
Tree Sizes:
[[  1.  -1.   0.   1. -inf -inf -inf]
 [ -1.   2.  -1.  -1.  -1. -inf -inf]
 [  1.   0.  -1.  -1.  -1.  -1. -inf]
 [ -1.  -1.  -1.  -1.  -1.  -1.  -1.]
 [-inf  -1.   1.   0.  -1.  -1.  -1.]
 [-inf -inf   2.   1.   0.  -1.  -1.]
 [-inf -inf -inf   3.  -1.   1.   0.]] 

```

This board shows you each board space and the size of the tree thereon. 

- -inf is an invalid board space
- -1. is am empty space
- 0 is a seed
- 1 is a small tree
- 2 is a medium tree
- 3 is a large tree

#### Player Occupancy Board:
```
Player Occupancies
[[  1.  -1.   1.   1. -inf -inf -inf]
 [ -1.   1.  -1.  -1.  -1. -inf -inf]
 [  2.   1.  -1.  -1.  -1.  -1. -inf]
 [ -1.  -1.  -1.  -1.  -1.  -1.  -1.]
 [-inf  -1.   0.   2.  -1.  -1.  -1.]
 [-inf -inf   0.   2.   0.  -1.  -1.]
 [-inf -inf -inf   2.  -1.   0.   0.]] 

```

This board shows you which players own which trees.

- -inf is an invalid board space
- -1 is an unclaimed space
- 0-3 mean the space is claimed by the respective player.

#### Possible Actions:

```
Please Choose Move From Available Options: 
Possible moves:
[0]: {'(pass_turn)'}
[1]: {"('buy_tree', size=0)"}
[2]: {"('buy_tree', size=1)"}
[3]: {"('buy_tree', size=2)"}
[4]: {"('plant_seed', pos=((1, 0), parent=(1, (2, 0))))"}
[5]: {"('plant_seed', pos=((3, 0), parent=(1, (2, 0))))"}
[6]: {"('plant_seed', pos=((3, 1), parent=(1, (2, 0))))"}
[7]: {"('plant_seed', pos=((6, 4), parent=(1, (5, 3))))"}
[8]: {"('plant_seed', pos=((3, 0), parent=(3, (6, 3))))"}
[9]: {"('plant_seed', pos=((3, 1), parent=(3, (6, 3))))"}
[10]: {"('plant_seed', pos=((3, 2), parent=(3, (6, 3))))"}
[11]: {"('plant_seed', pos=((3, 3), parent=(3, (6, 3))))"}
[12]: {"('plant_seed', pos=((4, 1), parent=(3, (6, 3))))"}
[13]: {"('plant_seed', pos=((4, 4), parent=(3, (6, 3))))"}
[14]: {"('plant_seed', pos=((5, 5), parent=(3, (6, 3))))"}
[15]: {"('plant_seed', pos=((6, 4), parent=(3, (6, 3))))"}
[16]: {"('grow_tree', size=0, pos=(4, 3))"}
[17]: {"('harvest_tree', pos=(6, 3))"}
Num Suns: 4
Input Choice:
```

This displays the available actions you have this turn, based on your inventory, the state of the board, and your number of remaining sun points.

Input the number for the action you want to choose.

Types of actions include:
- initial_placement: Available in setup phase of game only. Place a small tree on the board to start at pos=(i,j).
- pass_turn: Do nothing else. Turn passes on to the next player.
- buy_tree: Buy a tree of size={0,1,2, or 3} from your store, and put into your inventory.
- plant_seed: Move a seed (tree size 0) from your inventory onto an unoccupied space with pos=(i,j) on the board. 
    Seed uses action of parent tree (parent=(<parent_size>, (<parent_i>, <parent_j>))).
- grow_tree: Use a tree of size n+1 to grow a tree of size=n on the board at position pos=(i,j).
- harvest_tree: Harvest a tree of size=3 at board space with pos=(i,j) for points. 
