from .player import Player

from abc import ABC

class HumanPlayer(Player):
    """An abstract class representing an Human Player.
    
    Subclasses must implement the following method:

    - `make_move()`

    """
    def __init__(self, player_name=None, player_num=None):
        super(HumanPlayer, self).__init__(is_bot=False, player_name=player_name, player_num=player_num)

    def change_name(self, new_name):
        self.__name = new_name