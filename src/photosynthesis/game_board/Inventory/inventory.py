from src.Photosynthesis.GameBoard.Trees import Tree
from src.Photosynthesis.GameBoard.Exceptions import OutOfStockException

class PlayerInventory:
    def __init__(self, player_num):
        self.__player_num = player_num
        self.__starting_vals = {0:2, 1:4, 2:1, 3:0}
        self.__trees: dict[int, list[Tree]]

        self.reset()

    def reset(self):
        self.__trees = {}
        for tree_size in self.__starting_vals:
            self.__trees[tree_size] = [Tree(size=tree_size, player_num=self.__player_num) \
                                        for k in range(self.__starting_vals[tree_size])]

    def add_tree(self, tree:Tree):
        if tree.player == self.__player_num:
            self.__trees[tree.size].append(tree)
        else:
            raise ValueError('Tree does not belong to this player')

    def remove_tree(self, size):
        if len(self.__trees[size]) == 0:
            raise OutOfStockException(f"No trees of size {size} in Player {self.player_num}'s inventory")
        
        return self.__trees[size].pop(0)
        
    @property
    def player_num(self):
        return self.__player_num
    
    def num_available_trees(self, size):
        return len(self.__trees[size])

