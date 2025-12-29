from .ai.deep_q_learning import RLPlayer
from .human_player import PhotosynthesisHumanPlayer as HumanPlayer
from .ai.random_player import PhotosynthesisRandomPlayer as RandomPlayer
from .ai.deep_q_learning import RLPlayer

__all__ = ['AIPlayer', 'HumanPlayer', 'RandomPlayer']