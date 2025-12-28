import numpy as np
from collections import defaultdict
from src.Photosynthesis.GameBoard.Exceptions import OutOfStockException
from src.Photosynthesis.GameBoard.Trees import Tree

class PlayerStore:
    def __init__(self, player_num):
        self.__player_num = player_num
        self.__capacities = {0:4, 1:4, 2:3, 3:2}

        self.__inventory:dict[int, list[Tree]] = {}
        for tree_size in self.__capacities:
            self.__inventory[tree_size] = [Tree(size=tree_size, player_num=self.__player_num) \
                                            for k in range(self.__capacities[tree_size])]

        # initialize costs
        # self.__costs[i][n] is the cost if buying a tree of size i when n are available in the store
        self.__costs = {0:[np.inf, 2, 2, 1, 1],
                        1:[np.inf, 3, 3, 2, 2],
                        2:[np.inf, 4, 3, 3],
                        3:[np.inf, 5, 4]}
        
    
    # Public Functions
    def get_num_available(self, size):
        return self.__inventory[size]
    
    def get_cost(self, size):
        num_available = len(self.__inventory[size])
        return self.__costs[size][num_available]
    
    def buy_tree(self, size):
        if len(self.__inventory[size]) == 0:
            raise OutOfStockException(f'No trees available of size {size}')
        
        return self.__inventory[size].pop(0)

    def restock_tree(self, tree:Tree):
        if len(self.__inventory[tree.size]) < self.__capacities[tree.size]:
            self.__inventory[tree.size].append(tree)
            tree.clear_position()

    @property
    def player_num(self):
        return self.__player_num


