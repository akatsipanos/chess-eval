# %%
import argparse
import logging
import re
from pathlib import Path
from typing import Any, TypedDict
from dataclasses import dataclass

import chess
import numpy as np
from yaml import safe_load

from stockfish import Stockfish

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class MoveDict(TypedDict):
    move: str
    time: str

class MovesDict(TypedDict):
    white: MoveDict
    black: MoveDict | None

class GameDict(TypedDict):
    ratings: tuple[int, int]
    moves: list[MovesDict]
    result: int
    total_time: int


def split_games_new(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding='utf-8') as file:
        data = file.read()
    split_data = data.split('\n\n\n')
    split_data_lines = [sd.split('\n') for sd in split_data]
    return split_data, split_data_lines


def processing_new(path: Path) -> list[GameDict]:
    """
    [
    {"ratings": (1,2),
     "moves": [{"white": {"move": "e4",
                          "time": "5"} ,
                "black": {"move":"e5",
                          "time": "10"}},
                {}],
     "result": 1,
     "total_time": 7200
     },
     {"ratings":
      "moves":
      "results":
      "total_time":
     }
    ]
    """
    games, games_lines = split_games_new(path)
    output = []
    for game, game_lines in zip(games, games_lines):
        game_dict: GameDict = {}

        # Replace with appropriate regex
        ratings = (game.find('WhiteElo'), game.find("BlackElo"))
        result = game.find('Result')
        total_time = game.find('TimeControl').split('+')[0]
        ###

        game_dict['ratings'] = ratings
        game_dict['result'] = result
        game_dict['total_time'] = total_time

        # To get moves into desired format, need to convert the format - 1. d4 {[%emt 00:00:00]} d5 {[%emt 00:00:00]}
        # into the above json format. Basically, find a regex to get each component out
        moves = game_lines[-1]
        split_moves = ''
        game_dict["moves"] = []
        for move_pair in split_moves:
            move_pair_extracted = ''
            move_dict = {"white": {"move": move_pair_extracted[0][0],
                                   "time": move_pair_extracted[0][1]},
                         "black": {"move": move_pair_extracted[1][0],
                                   "time": move_pair_extracted[1][1]}}
            game_dict["moves"].append(move_dict)


        output.append(game_dict)


    return output


@dataclass
class DataProcessing:
    sf_depth = 10
    sf_path = ''
    data_path = Path('')
    row_count = 0
    output_path = Path('')
    verbose = 500
    data_type = 'train, val, test'

    def load_data(self):
        with open(self.data_path, encoding='utf-8') as file:
            data = file.read()
        split_data = data.split('\n\n\n')
        split_data_lines = [sd.split('\n') for sd in split_data]
        return split_data, split_data_lines

    def process_data(self):
        pass

    def create_rows(self):
        pass

    def generatre_matrix(self, save_data = True):
        pass

    def filter_moves(self):
        moves = 


def main() -> None:
    logging.debug("initialising processing...")
    base_dir = Path(__file__).parent.parent.resolve()
    data_dir = base_dir / "raw"
    output_dir = data_dir / "processed"

    with open(base_dir/"config.yml", 'r') as config_file:
        config = safe_load(config_file)

    raw_data_path = data_dir / config["raw_data_path"]
    output_path = output_dir / config["output_path"]
    stockfish_path = base_dir / config['sf_path']

    processing = DataProcessing()
    output_matrix = processing.generatre_matrix()


    ratings, moves, results, game_times = processing(raw_data_path)
    logging.info("total number of games: %s", len(moves))
    rows = count_rows(moves)
    logging.info("total number of rows: %s", rows)
    logging.info("creating final dataset...")
    output_final = create_final_matrix(
        rows,
        moves,
        ratings,
        game_times,
        results,
        stockfish_depth,
        verbose,
        stockfish_path
    )

    save_data(output_path/ "train.npy", output_final)

    return output_final


if __name__ == "__main__":
    output = main()

# Need to sort out the output. Final output should be an array of shape (452215,70)


# %%
