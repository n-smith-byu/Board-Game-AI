from abc import ABC, abstractmethod
from .players import Player, AIPlayer, HumanPlayer
from .exceptions import TooManyPlayersException

from typing import Type

class BoardGame(ABC):

    @classmethod
    @abstractmethod
    def get_min_num_players(cls) -> int:
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def get_max_num_players(cls) -> int:
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def get_ai_player_class(cls) -> Type[AIPlayer]:
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def get_human_player_class(cls) -> Type[HumanPlayer]:
        raise NotImplementedError()
    
    # - - - - - 
    
    def __init__(self, players=None):
        if players is None:
            self._players: list[Player] = []
        else:
            self._players = players
    
    def add_players(self, players: list[Player]):
        if len(players) > BoardGame.get_max_num_players():
            raise TooManyPlayersException(BoardGame.get_max_num_players())
        
        