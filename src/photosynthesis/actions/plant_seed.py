from ._action import Action
from ..GameBoard.Trees import Tree

class PlantSeed(Action):
    def __init__(self, player_num, parent_tree:Tree, board_space:tuple):
        super(PlantSeed, self).__init__(player_num)
        self.__parent_tree = parent_tree
        self.__position = board_space

    @property
    def position(self):
        return self.__position
    
    @property
    def parent_tree(self):
        return self.__parent_tree
    
    def __str__(self):
        return f"('plant_seed', pos=({self.__position}, parent=({self.__parent_tree.size}, {self.__parent_tree.position})))"
    
    def sort_key(self):
        return (super().sort_key(), 1, self.__parent_tree.position, self.__position)