from abc import ABC, abstractmethod

class Player(ABC):

    def __init__(self, is_bot, player_name=None, player_num=None):
        self._name = player_name
        self._player_num = player_num
        self._is_bot = is_bot

    def _derive_name(self):
        return self._name if self._name is not None else f'Player_{self._player_num}'

    def is_bot(self):
        return self._is_bot
    
    def assign_player_num(self, player_num):
        """Method for setting a player number. Once assigned, it cannot be changed"""
        if self.player_num is None:
            self._player_num = player_num
            return True
        
        return False

    @property
    def player_num(self):
        return self._player_num
    
    @property
    def player_name(self):
        return self._derive_name()

    @abstractmethod
    def choose_move(self, game_board):
        raise NotImplementedError("Subclasses must implement this method")

