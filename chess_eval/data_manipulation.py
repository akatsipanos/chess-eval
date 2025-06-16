# %%
import argparse
import logging
import re
from pathlib import Path
from typing import Any, TypedDict
from dataclasses import dataclass

import chess
import numpy as np

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

def split_games(path: Path) -> list[list[str]]:
    with open(path) as file:
        data = file.readlines()

    games: list[list[str]] = []
    current_game: list[str] = []

    for line in data:
        line = line.strip()
        if line.startswith("[Event"):
            if current_game:
                games.append(current_game)
                current_game = []
        current_game.append(line)

    if current_game:
        games.append(current_game)

    return games

def split_games_new(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding='utf-8') as file:
        data = file.read()
    split_data = data.split('\n\n\n')
    split_data_lines = [sd.split('\n') for sd in split_data]
    return split_data, split_data_lines


def processing(
    path: Path,
) -> tuple[list[list[Any]], list[list[str]], dict[str, int], int]:
    games = split_games(path)
    games = [x for x in games][:1000]
    ratings = [[int(x[5][-6:-2]), int(x[6][-6:-2])] for x in games]
    moves_games = [x[-3] for x in games]
    moves = [re.split(r"\d\.|\d\d\.", x[: x.find("} {") + 2])[1:] for x in moves_games]
    result_map = {"1-0": 0, "1/2": 1, "0-1": 2}  # 0 white win, 1 draw, 2 black win
    results = [result_map[x[-3:]] for x in moves_games]
    game_times = 120 * 60
    return ratings, moves, results, game_times


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

def create_rows(
    move_list: list[str], total_time: int, ratings: list[int], results, depth, stockfish_path: Path
) -> list[np.array]:
    board = chess.Board()
    output = []
    wrt, brt = total_time, total_time
    sf = Stockfish(stockfish_path, depth=depth)

    for move_time in move_list:
        times = re.findall(r"\{[^}]+\}", move_time)
        time_strings = [x[-10:-2] for x in times]

        moves = (
            (move_time.split(" ")[1], move_time.split(" ")[-4])
            if len(move_time) > 25
            else (move_time.split(" ")[1], 0)
        )

        for i, move in enumerate(moves):
            try:
                if move == 0:
                    continue
                else:
                    if i == 0:
                        wrt -= time_to_seconds(time_strings[i])
                        turn = 0
                    elif i == 1:
                        brt -= time_to_seconds(time_strings[i])
                        turn = 1

                    fen = board.fen()  # Get the fen for the position
                    sf.set_fen_position(fen)  # For stockfish evaluation

                    features = [
                        convert_fen_to_matrix(fen),
                        wrt / total_time,
                        brt / total_time,
                        sf.get_evaluation()["value"] / 100,
                        turn,
                        ratings[0],
                        ratings[1],
                        results,
                    ]
                    inner_array = features[0]
                    remaining_elements = features[1:]
                    final_array = np.array(list(inner_array) + remaining_elements)
                    output.append(final_array)

                    board.push_san(move)
            except KeyError:
                continue
    return output


def time_to_seconds(time_str: str) -> float:
    hours, minutes, seconds = map(int, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def convert_fen_to_matrix(fen_string: str) -> np.ndarray[tuple[int], np.dtype[Any]]:
    piece_values = {
        "P": 0.1,
        "N": 0.3,
        "B": 0.35,
        "R": 0.5,
        "Q": 0.9,
        "K": 1.0,
        "p": -0.1,
        "n": -0.3,
        "b": -0.35,
        "r": -0.5,
        "q": -0.9,
        "k": -1.0,
    }

    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(8)] for _ in range(8)]

    # Split FEN string into relevant parts
    fen_parts = fen_string.split()
    board_part = fen_parts[0]
    ranks = board_part.split("/")

    # Iterate over ranks and files to fill the matrix
    for rank, fen_rank in enumerate(ranks):
        file_index = 0
        for char in fen_rank:
            if char.isdigit():
                # Empty squares
                file_index += int(char)
            else:
                # Piece squares
                matrix[rank][file_index] = piece_values[char]
                file_index += 1

    return np.array(matrix).flatten()


def count_rows(moves: list[list[str]]) -> int:
    rows = 0
    for move in moves:
        for i in move:
            if len(i) > 25:
                rows += 2
            else:
                rows += 1
    return rows

def count_rows_new(json_data: list[GameDict]) -> int:
    """
    Just cycle through all the games and count the number of keys in each MoveDict
    """
    rows = 0
    for game_dict in json_data:
        for move in game_dict["moves"]:
            rows += len(move.keys()) # 1 key means 1 move, 2 keys means 2 moves

def create_final_matrix(
    rows: int,
    moves: list[list[str]],
    ratings: list[list[int]],
    game_times: int,
    results: dict[str, int],
    depth: int,
    verbose: int,
    stockfish_path: Path
) -> np.ndarray:
    output_final = np.ndarray((rows, 71))
    i = 0
    for moves_in_each_game, rating, result in zip(moves, ratings, results):
        fen_list = create_rows(moves_in_each_game, game_times, rating, result, depth, stockfish_path)
        for j in fen_list:
            output_final[i] = j
            i += 1
            if i % verbose == 0 or i == rows - 1:
                logging.info("Processed %s rows", i)
    return output_final


def save_data(path: Path, data) -> None:
    np.save(file=path, arr=data)

@dataclass
class DataProcessing:
    sf_depth = 10
    sf_path = ''
    data_path = Path('')
    row_count = 0
    output_path = Path('')
    verbose = 500

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

    def generatre_matrix(self):
        pass

    def filter_moves(self):
        moves = 


def main() -> None:
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "-o", "--output-path", type=str, required=True, help="output dataset path"
        )
        ap.add_argument(
            "-r", "--raw-data-path", type=str, required=True, help="path to raw data"
        )
        ap.add_argument(
            "-v",
            "--verbose",
            default=500,
            type=int,
            help="number of rows processed before print statement",
        )
        ap.add_argument(
            "-d",
            "--stockfish-depth",
            default=10,
            type=int,
            help="depth of stockfish engine evaluation",
        )
        args = ap.parse_args()
        output_data_path_dir = args.output_path
        raw_data_file_name = args.raw_data_path
        stockfish_depth = args.stockfish_depth
        verbose = args.verbose
    except SystemExit:
        output_data_path_dir = "output"
        raw_data_file_name = "2018_03_11-2023_05_25.pgn"
        stockfish_depth = 10
        verbose = 500

    logging.info("initial processing...")
    base_dir = Path(__file__).parent.parent.resolve()
    data_dir = base_dir / "data"
    raw_data_path = data_dir / "games" / "train" / raw_data_file_name
    output_dir_path = data_dir / output_data_path_dir
    output_data_file_name = "train.npy"
    stockfish_path = base_dir / "stockfish" / "stockfish-windows-x86-64-avx2.exe"


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

    save_data(output_dir_path/ output_data_file_name, output_final)

    return output_final


if __name__ == "__main__":
    output = main()

# Need to sort out the output. Final output should be an array of shape (452215,70)


# %%
