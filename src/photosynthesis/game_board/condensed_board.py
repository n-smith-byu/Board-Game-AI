from .game_board import GameBoard
from src.Photosynthesis.ActionTypes import InitialPlacement

class BoardSummary:
    def __init__(self, player_num, game_board:GameBoard, remaining_turns:int,
                 total_game_turns, possible_actions, init_setup=False):
        
        self.player_num = player_num
        self.player_positions = game_board.get_player_board()
        self.tree_board = game_board.get_tree_board()
        self.sun_pos = game_board.get_sun_pos()
        self.total_game_turns = total_game_turns
        self.remaining_turns = remaining_turns
        self.player_suns = game_board.get_player_suns(player_num)
        self.player_new_suns = game_board.get_player_new_suns_this_turn(player_num)
        self.player_score = game_board.get_player_score(player_num)
        self.init_setup = init_setup

        self.available_actions = possible_actions.copy()