class Tree:
    COST_TO_PLACE = {0:1, 1:1, 2:2, 3:3}
    COST_TO_GROW = {0:1, 1:2, 2:3, 3:4}
    def __init__(self, size:int, player_num:int, pos:tuple=None):
        self.__size = size
        self.__pos = pos
        self.__player_num = player_num

        self.__grown_flag = False
    
    def __hash__(self):
        return hash((*self.__pos, self.__size, self.__player_num))
    
    def cost_to_grow(self):
        return Tree.COST_TO_GROW[self.__size]
    
    def cost_to_place(self):
        return Tree.COST_TO_PLACE[self.size]
    
    def set_position(self, pos):
        self.__pos = pos

    def clear_position(self):
        self.__pos = None

    def clear_grown_flag(self):
        self.__grown_flag = False

    def set_grown_flag(self):
        self.__grown_flag = True

    def grown_this_turn(self):
        return self.__grown_flag
    
    @property
    def size(self):
        return self.__size
    
    @property
    def position(self):
        return self.__pos
    
    @property
    def player(self):
        return self.__player_num