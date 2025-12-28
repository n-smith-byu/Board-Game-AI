from ._action import Action

class PassTurn(Action):
    def __init__(self, player_num):
        super(PassTurn, self).__init__(player_num)

    def __str__(self):
        return "(pass_turn)"
    
    def sort_key(self):
        return (super().sort_key(), -2)