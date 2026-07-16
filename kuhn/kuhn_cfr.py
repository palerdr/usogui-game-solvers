from enum import Enum, IntEnum

class Card(IntEnum):
    Jack = 0
    Queen = 1
    King = 2

Deck = {Card.Jack, Card.Queen, Card.King}

class Action(IntEnum):
    Check = 0
    Bet = 1
    Call = 2
    Fold = 3



