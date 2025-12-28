from ._action import Action
from src.Photosynthesis.GameBoard.Trees import Tree

class HarvestTree(Action):
    def __init__(self, player_num, tree:Tree):
        super(HarvestTree, self).__init__(player_num)

        if player_num != tree.player:
            raise ValueError('Cannot harvest tree of another player')
    
        self.__tree = tree

    @property
    def tree(self):
        return self.__tree
    
    def __str__(self):
        return f"('harvest_tree', pos={self.__tree.position})"
    
    def sort_key(self):
        return (super().sort_key(), 3, self.__tree.position)
