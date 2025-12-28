from ..Players import Player, HumanPlayer, AIPlayer
from ..board_game import BoardGame
from ..Exceptions import NotEnoughPlayersException, TooManyPlayersException

from typing import Type
import random
import uuid


class WaitingRoom:
    def __init__(self, GameClass: Type[BoardGame]):
        
        self.game_class = GameClass
        self.human_player_class: Type[HumanPlayer] = GameClass.get_human_player_class()
        self.ai_player_class: Type[AIPlayer] = GameClass.get_ai_player_class()

        self.__num_bots: int = 0
        self.__num_humans: int = 0

        self.__players: list[Player] = []
        self.__player_keys: dict[str, Player] = {}
        
    def add_player(self, player_name=None, **kwargs):
        if len(self.__players) >= self.game_class.get_max_num_players():
            raise TooManyPlayersException(self.game_class.get_max_num_players())
        
        unique_key = str(uuid.uuid4())
        self.__players.append(self.human_player_class(player_name=player_name))
        self.__player_keys[unique_key] = self.__players[-1]
        self.__num_humans += 1

        return unique_key

    def remove_player(self, unique_key) -> HumanPlayer:
        """Removes a Human Player from the list if they exist"""
        self.__players.remove(self.__player_keys[unique_key])
        player = self.__player_keys.pop(unique_key)
        self.__num_humans -= 1

        return player
            
    def add_bot(self):        
        if len(self.__players) >= self.game_class.get_max_num_players():
            raise TooManyPlayersException(self.game_class.get_max_num_players())
        
        self.__players.append(self.ai_player_class())
        self.__num_bots += 1
        return True

    def remove_bot(self):
        """Removes a a bot if there are any in the player list. """
        bots = [player for player in self.__players if player.is_bot()]
        if len(bots) > 0:
            self.__players.remove(bots[-1])
            self.__num_bots -= 1
            return True
        
        return False
    
    def get_player_list(self):
        players = []
        for key, player in self.__player_keys.items():
            players.append((player.player_name, key))

        for i in range(self.__num_bots):
            players.append((f'Bot{i + 1}'))

        return players
    
    def total_num_players(self):
        return len(self.__players)
    
    def num_human_players(self):
        return self.__num_humans
    
    def num_bots(self):
        return self.__num_bots

    def create_game(self, **game_parameters):
        if len(self.__players) < self.game_class.get_min_num_players():
            raise NotEnoughPlayersException()
        for i, player in enumerate(self.__players):
            player.assign_player_num(i)
        return self.game_class(players=self.__players, **game_parameters)
