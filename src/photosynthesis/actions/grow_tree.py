from ._action import Action
from ..game_board import Tree

class GrowTree(Action):
    def __init__(self, player_num, tree:Tree):
        super(GrowTree, self).__init__(player_num)
        if tree.size >= 3:
            raise ValueError('Tree of size 3 cannot be grown. Please use HarvestTree action instead.')
        
        self.__tree = tree

    @property
    def tree(self):
        return self.__tree
    
    def __str__(self):
        return f"('grow_tree', size={self.__tree.size}, pos={self.__tree.position})"
    
    def sort_key(self):
        return (super().sort_key(), 2, self.__tree.size, self.__tree.position)
