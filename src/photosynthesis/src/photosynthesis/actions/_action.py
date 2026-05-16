from abc import ABC, abstractmethod

class Action(ABC):
    def __init__(self, player_num):
        self._player = player_num

    @property
    def player(self):
        return self._player
    
    @abstractmethod
    def __str__(self):
        return str(self._player)
    
    @abstractmethod
    def sort_key(self):
        return self._player
        