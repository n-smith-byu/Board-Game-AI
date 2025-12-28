from src.BoardGame.Players import AIPlayer
from src.Photosynthesis.GameBoard import BoardSummary
import numpy as np

class PhotosynthesisRandomPlayer(AIPlayer):
    def __init__(self, player_num=None):
        super(PhotosynthesisRandomPlayer, self).__init__(player_num)

    def choose_move(self, state:BoardSummary):
        # chooses a random move
        return np.random.choice(len(state.available_actions))