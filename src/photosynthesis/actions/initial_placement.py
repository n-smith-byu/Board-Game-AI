from ._action import Action

class InitialPlacement(Action):
    def __init__(self, player_num, position:tuple[int], ind):
        super(InitialPlacement, self).__init__(player_num)

        self.__position = position
        self.__ind = ind
    
    @property
    def position(self):
        return self.__position
    
    def __str__(self):
        return f"('initial_placement', pos={self.__position})"
    
    def sort_key(self):
        return (super().sort_key(), -1, self.__ind)