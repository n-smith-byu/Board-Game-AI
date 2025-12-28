
class InvalidPlayerCountException(Exception):
    def __init__(self, message: str):
        super(InvalidPlayerCountException, self).__init__(message)

class NotEnoughPlayersException(InvalidPlayerCountException):
    def __init__(self, min_player_count: int):
        message = f"Not enough players. Min of {min_player_count} players required."
        super(NotEnoughPlayersException, self).__init__(message)

class TooManyPlayersException(InvalidPlayerCountException):
    def __init__(self, max_player_count: int):
        message = f"Too many players. Max of {max_player_count} players allowed."
        super(TooManyPlayersException, self).__init__(message)

    