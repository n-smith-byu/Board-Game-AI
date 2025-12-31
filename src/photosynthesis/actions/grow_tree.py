from ._action import Action
from ..game_board import Tree

class GrowTree(Action):
    def __init__(self, player_num, tree:Tree):
        super(GrowTree, self).__init__(player_num)
        if tree.size >= 3:
            raise ValueError('Tree of size 3 cannot be grown. Please use HarvestTree action instead.')
        
        self._tree = tree

    @property
    def tree(self):
        return self._tree
    
    def __str__(self):
        return f"('grow_tree', size={self._tree.size}, pos={self._tree.position})"
    
    def sort_key(self):
        return (super().sort_key(), 2, self._tree.size, self._tree.position)
