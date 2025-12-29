from .player import Player

from abc import ABC

class AIPlayer(Player):
    """An abstract class representing an AI Player.

    Subclasses must implement the following method from Player:
    - `choose_move()`

    """
    def __init__(self, player_num=None):
        super(AIPlayer, self).__init__(is_bot=True, player_name=None, player_num=player_num)

    def _derive_name(self):
        return f'AI_Player_{self._player_num}'
