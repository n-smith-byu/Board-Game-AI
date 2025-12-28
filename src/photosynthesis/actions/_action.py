from abc import ABC, abstractmethod

class Action(ABC):
    def __init__(self, player_num):
        self.__player = player_num

    @property
    def player(self):
        return self.__player
    
    @abstractmethod
    def __str__(self):
        return str(self.__player)
    
    @abstractmethod
    def sort_key(self):
        return self.__player
        