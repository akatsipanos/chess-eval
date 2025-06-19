from typing import TypedDict


class MoveDict(TypedDict, total=False):
    move: str
    time: int


class MovesDict(TypedDict, total=False):
    white: MoveDict
    black: MoveDict


class GameDict(TypedDict, total=False):
    ratings: tuple[int, int]
    moves: list[MovesDict]
    result: int
    total_time: int
