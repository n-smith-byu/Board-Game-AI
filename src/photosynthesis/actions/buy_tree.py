from ._action import Action

class BuyTree(Action):
    def __init__(self, player_num, size):
        super(BuyTree, self).__init__(player_num)
        self.__size = size

    @property
    def size(self):
        return self.__size
    
    def __str__(self):
        return f"('buy_tree', size={self.__size})"
    
    def sort_key(self):
        return (super().sort_key(), 0, self.__size)