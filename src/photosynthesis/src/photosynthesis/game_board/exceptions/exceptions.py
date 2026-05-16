class SpaceUnavailableException(ValueError):
    def __init__(self, message):
        super(SpaceUnavailableException, self).__init__(message)

class InvalidBoardSpaceException(ValueError):
    def __init__(self, pos:tuple, valid_board_spaces:set):
        message = f'{pos} id not a valid board space. Valid board spaces include {valid_board_spaces}.'
        super(InvalidBoardSpaceException, self).__init__(message)

class InsufficientSunsException(Exception):
    def __init__(self, message):
        super(InsufficientSunsException, self).__init__(message)

class OutOfStockException(Exception):
    def __init__(self, message):
        super(OutOfStockException, self).__init__(message)

class TreeAlreadyUsedActionException(Exception):
    def __init__(self, message):
        super(TreeAlreadyUsedActionException, self).__init__(message)


