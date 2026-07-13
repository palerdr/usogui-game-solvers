from enum import Enum
from typing import List
from poker_math import CardMask
from dataclasses import dataclass

class Action(Enum):
    Check = 0
    Bet = 1
    Call = 2
    Raise = 3
    Fold = 4
    AllIn = 5


@dataclass
class RoundState:
    

def generate_legal_hands(n: int, mask: CardMask ) -> List[CardMask]: ...
"""Given an integer n generate all the hands given the current
card mask of used cards in the deck, all of the card combinations
satisfying \(\sum_{c\in H} v(c)=n,\). Suits do not alter the numerical sum,
but they determine whether a hand is a flush or straight flush"""

def in_safe_zone(xi: int, yi: int, antes_remaining: int) -> bool:
    """A player has a lead so great the remaining antes can be lost 
    and they can still outlast their opponent """
    return xi - antes_remaining > yi + antes_remaining