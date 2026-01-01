from src.board_game.players import AIPlayer
from src.photosynthesis.game import PlayerGameState

import numpy as np

class PhotosynthesisRandomPlayer(AIPlayer):
    def __init__(self, player_num=None):
        super(PhotosynthesisRandomPlayer, self).__init__(player_num)

    def choose_move(self, state: PlayerGameState):
        # chooses a random move
        return np.random.choice(len(state.available_actions))