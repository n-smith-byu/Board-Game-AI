from .photosynthesis_game import GameBoard
from ..actions import *

class PlayerGameState:
    def __init__(self, player_num, num_players, first_player_token, 
                 game_board: GameBoard, remaining_turns:int,
                 total_game_turns, possible_actions, 
                 initial_setup=False):
        
        self.player_num = player_num
        self.num_players = num_players
        self.first_player_token = first_player_token

        self.player_positions = game_board.get_player_board()
        self.tree_board = game_board.get_tree_board()

        self.player_suns = game_board.get_player_suns(player_num)
        self.player_points = game_board.get_player_score(player_num)
        self.player_new_suns = game_board.get_player_new_suns_this_turn(player_num)
        self.player_stores = game_board.get_player_stores()
        self.player_inventories = game_board.get_player_inventories()
        self.all_player_suns = [game_board.get_player_suns(ind) for ind in range(num_players)]
        self.all_player_points = [game_board.get_player_score(ind) for ind in range(num_players)]

        self.sun_pos = game_board.get_sun_pos()
        self.total_game_turns = total_game_turns
        self.remaining_turns = remaining_turns

        self.available_actions = possible_actions.copy()
        self.initial_setup = initial_setup
