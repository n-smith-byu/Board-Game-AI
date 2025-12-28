from abc import ABC, abstractmethod

class GameBoard(ABC):
    def __init__(self, num_players: int):
        self.__num_players = num_players

    def possible_moves(self, player_num: int):
        if player_num < 0 or player_num >= self.__num_players:
            raise ValueError(f'Invalid Player Number for a game with {self.__num_players} players')
        
        self.__get_possible_moves(player_num)

    @abstractmethod
    def __get_possible_moves(self, player_num: int):
        raise NotImplementedError('Subclasses must implement this method')