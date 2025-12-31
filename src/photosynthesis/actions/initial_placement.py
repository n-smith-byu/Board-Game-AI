from ._action import Action

class InitialPlacement(Action):
    def __init__(self, player_num, position:tuple[int], ind):
        super(InitialPlacement, self).__init__(player_num)

        self._position = position
        self._ind = ind
    
    @property
    def position(self):
        return self._position
    
    def __str__(self):
        return f"('initial_placement', pos={self._position})"
    
    def sort_key(self):
        return (super().sort_key(), -1, self._ind)