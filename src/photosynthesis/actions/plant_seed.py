from ._action import Action
from ..game_board.trees import Tree

class PlantSeed(Action):
    def __init__(self, player_num, parent_tree:Tree, board_space:tuple):
        super(PlantSeed, self).__init__(player_num)
        self._parent_tree = parent_tree
        self._position = board_space

    @property
    def position(self):
        return self._position
    
    @property
    def parent_tree(self):
        return self._parent_tree
    
    def __str__(self):
        return f"('plant_seed', pos=({self._position}, parent=({self._parent_tree.size}, {self._parent_tree.position})))"
    
    def sort_key(self):
        return (super().sort_key(), 1, self._parent_tree.position, self._position)