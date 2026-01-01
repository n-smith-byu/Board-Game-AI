from ..players import RLPlayer, HumanPlayer
from ..game_board import GameBoard
from .game_state import PlayerGameState
from ..actions import BuyTree, PlantSeed, GrowTree, HarvestTree, PassTurn, InitialPlacement
from src import board_game as bg

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...board_game.players import Player

class PhotosynthesisGame(bg.BoardGame):
    @classmethod
    def get_max_num_players(cls) -> int:
        return 4
    
    @classmethod
    def get_min_num_players(cls) -> int:
        return 2
    
    @classmethod
    def get_ai_player_class(cls) -> type[bg.players.AIPlayer]:
        return RLPlayer
    
    @classmethod
    def get_human_player_class(cls) -> type[bg.players.HumanPlayer]:
        return HumanPlayer

    SUN_POSITIONS = 6

    def __init__(self, players: list[bg.players.Player], 
                 extra_round: bool=False):
        super(PhotosynthesisGame, self).__init__(players)
        self._num_players = len(players)
        self._extra_round = extra_round

        self.reset()

    def reset(self):
        self.board = GameBoard(self._num_players)

        self._num_rounds = 3 + self._extra_round
        self._current_round = -1
        self._first_player_token = 0
        self._curr_player = 0
        self._game_over = False

    # Public Methods

    def run(self, display=False):
        available_spaces = self.board.get_possible_first_turn_spaces()
        # setting initial trees on board
        for i in range(2):
            self._curr_player = self._first_player_token
            for i in range(self._num_players):
                player_num = self._curr_player
                player: Player = self._players[player_num]

                possible_actions = [InitialPlacement(player_num, pos, i) \
                                                       for i, pos in enumerate(available_spaces)]
                
                game_state = PlayerGameState(
                    player_num,
                    self._num_players,
                    self._first_player_token, 
                    self.board,
                    remaining_turns=self._num_rounds*6 + 2 - i,
                    possible_actions=possible_actions,
                    total_game_turns=self._num_rounds*6 + 2,
                    initial_setup=True
                    )

                while True:
                    if display:
                        print(f"{player.player_name}'s Turn")
                        self.board.print_boards()
                        
                    action_ind = player.choose_move(game_state)
                    try:
                        action:InitialPlacement = possible_actions[action_ind]
                    except Exception as ex:
                        print('Invalid Action')
                        continue

                    self.board.player_initial_tree_placement(action.player, action.position)
                    available_spaces.remove(action.position)
                    break

                self._curr_player = (self._curr_player + 1) % self._num_players

        # playing the game
        self.board.rotate_sun()
        for round in range(self._num_rounds):
            self._current_round = round

            for sun_pos in range(PhotosynthesisGame.SUN_POSITIONS):
                self._curr_player = self._first_player_token
                if display:
                    print(f'Sun Pointing: {self.board.get_sun_direction_vec()}')
                for i in range(self._num_players):
                    player: Player = self._players[self._curr_player]
                    while True:      # until player passes their turn

                        possible_actions = self.board.get_possible_actions(self._curr_player)
                        game_state = PlayerGameState(
                            self._curr_player, 
                            self._num_players,
                            self._first_player_token,
                            self.board,
                            remaining_turns = (self._num_rounds - round)*6 - sun_pos,
                            total_game_turns=self._num_rounds*6 + 2,
                            possible_actions=possible_actions,
                            initial_setup=False
                            )
            
                        while True:     # until valid action chosen
                            if display:
                                print(f"{player.player_name}'s Turn")
                                self.board.print_boards()

                            action_ind= player.choose_move(game_state)
                            try:
                                action = possible_actions[action_ind]
                            except Exception as ex:
                                print('Invalid Action')
                                continue
                            else:
                                break

                        if not isinstance(action, PassTurn):
                            self.apply_action(action)
                        else:
                            break
                            
                    
                    self._curr_player = (self._curr_player + 1) % self._num_players
                    
                self._first_player_token = (self._first_player_token + 1) % self._num_players
                if not (round == self._num_rounds and self.board.get_sun_pos() == 5): 
                    self.board.rotate_sun()
            
        self._game_over = True
        scores = self.board.get_player_scores()
        for player_num in range(self._num_players):
            remaining_suns = self.board.get_player_suns(player_num)
            scores[player_num] += remaining_suns // 3

        return tuple(scores)

    # Getters
    def get_num_rounds(self):
        return self._num_rounds

    def get_current_round(self):
        return self._current_round
    
    def get_first_player(self):
        player = self._players[self._first_player_token]
        return (player.player_num, player.player_name)
    
    def get_sun_pos(self):
        return self.board.get_sun_pos()
    
    def game_ended(self):
        return self._game_over
    
    def apply_action(self, action):
        if isinstance(action, BuyTree):
            self.board.player_buy_tree(action.player, action.size)
        elif isinstance(action, PlantSeed):
            self.board.player_plant_seed(action.player, action.parent_tree, action.position)
        elif isinstance(action, GrowTree):
            self.board.player_grow_tree(action.player, action.tree)
        elif isinstance(action, HarvestTree):
            self.board.player_harvest_tree(action.player, action.tree)
        elif isinstance(action, PassTurn):
            pass