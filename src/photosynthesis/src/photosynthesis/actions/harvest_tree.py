from ._action import Action

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..game_board import Tree

class HarvestTree(Action):
    def __init__(self, player_num, tree: 'Tree'):
        super(HarvestTree, self).__init__(player_num)

        if player_num != tree.player:
            raise ValueError('Cannot harvest tree of another player')
    
        self._tree = tree

    @property
    def tree(self):
        return self._tree
    
    def __str__(self):
        return f"('harvest_tree', pos={self._tree.position})"
    
    def sort_key(self):
        return (super().sort_key(), 3, self._tree.position)
