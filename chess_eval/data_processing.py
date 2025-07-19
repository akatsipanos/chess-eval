# %%
import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
from numpy.typing import NDArray
from stockfish import Stockfish
from yaml import safe_load

from chess_eval.constants import BASE_DIR, SF_PATH
from chess_eval.schemas import GameDict, MovesDict
from chess_eval.utils import convert_fen_to_matrix, convert_time_str_to_seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# TODO
# - Add better logging throughout
# - Fix up the init vars
# -
@dataclass
class DataProcessing:
    raw_data_path: Path
    output_dir: Path
    data_type: str
    sf_depth: int = field(default=10)
    moves_to_skip: int = 3
    number_of_games: int = 50
    verbose: int = 500

    row_count: int = field(init=False)
    json_data: list[GameDict] = field(init=False)

    def load_data(self) -> list[str]:
        with open(self.raw_data_path, encoding="utf-8") as file:
            data = file.read()
        split_data = data.split("\n\n\n")
        if self.number_of_games and self.number_of_games < len(split_data):
            split_data = split_data[: self.number_of_games]
        return split_data

    def process_raw_data(self) -> None:
        games = self.load_data()
        output = []
        for ind, game in enumerate(games):
            game_dict: GameDict = {}

            # Replace with appropriate regex
            # ratings = (game.find('WhiteElo'), game.find("BlackElo"))
            white_rating_match = re.search(r'\[WhiteElo "(\d.+)"\]', game)
            black_rating_match = re.search(r'\[BlackElo "(\d.+)"\]', game)
            result_match = re.search(r'\[Result "(\d.+)"\]', game)
            total_time_match = re.search(r'\[TimeControl "(\d.+)"\]', game)

            if not all(
                [white_rating_match, black_rating_match, result_match, total_time_match]
            ):
                logging.warning("Extracted values not found, skipping game - %s", ind)
                continue

            # findall returns list so take [0] values for everything
            # out of index error not possible as empty list picked up in
            # error handling
            white_rating = white_rating_match.group(1)  # type: ignore
            black_rating = black_rating_match.group(1)  # type: ignore
            ratings = (int(white_rating), int(black_rating))
            result_map = {"1-0": 0, "1/2": 1, "1/2-1/2": 1, "0-1": 2}
            result = result_map.get(result_match.group(1), 1)  # type: ignore
            game_dict["ratings"] = ratings
            game_dict["result"] = result
            game_dict["total_time"] = int(total_time_match.group(1).split("+")[0])  # type: ignore

            # To get moves into desired format, need to convert the format - 1. d4 {[%emt 00:00:00]} d5 {[%emt 00:00:00]}
            # into the above json format. Basically, find a regex to get each component out
            move_match = re.search(r"\n\n(1\..+)", game, re.DOTALL)
            if not move_match:
                logging.warning("Game %s skipped - no moves found", ind)
                continue
            move_str = move_match.group(1)

            pattern = r"([^\s{]+)\s+\{([^}]+)\}(?:\s+([^\s{]+)\s+\{([^}]+)\})?"
            split_moves = re.findall(pattern, move_str)
            game_dict["moves"] = []
            for split_move in split_moves:
                move_dict: MovesDict = {
                    "white": {
                        "move": split_move[0],
                        "time": convert_time_str_to_seconds(split_move[1]),
                    }
                }
                if all(split_move):
                    # try:
                    move_dict["black"] = {}
                    move_dict["black"]["move"] = split_move[2]
                    move_dict["black"]["time"] = convert_time_str_to_seconds(
                        split_move[3]
                    )
                    # except ValueError:
                    #     raise ValueError(f'{split_move}')
                game_dict["moves"].append(move_dict)

            output.append(game_dict)
        self.json_data = output
        with open(self.output_dir / f"{self.data_type}.json", "w") as file:
            json.dump(self.json_data, file)

    def _create_rows(self, game: GameDict) -> list[NDArray[np.float64]]:
        board = chess.Board()
        output = []
        # wrt, brt = total_time, total_time
        sf = Stockfish(SF_PATH, depth=self.sf_depth)

        total_time = game["total_time"]
        wrt, brt = total_time, total_time

        for move_pair in game["moves"]:
            # Step through each set of moves
            for color, move_dict in move_pair.items():
                # Step through white then black moves
                if color == "white":
                    wrt -= move_dict["time"]  # type:ignore
                    turn = 0
                elif color == "black":
                    brt -= move_dict["time"]  # type:ignore
                    turn = 1

                fen = board.fen()  # Get the fen for the position
                sf.set_fen_position(fen)  # For stockfish evaluation

                inner_array = convert_fen_to_matrix(fen)
                remaining_elements = [
                    wrt / total_time,
                    brt / total_time,
                    sf.get_evaluation()["value"] / 100,
                    turn,
                    game["ratings"][0],
                    game["ratings"][1],
                    game["result"],
                ]
                features = np.array(inner_array.tolist() + remaining_elements)
                output.append(features)

                board.push_san(move_dict["move"])  # type: ignore

        return output

    def _count_rows(self) -> None:
        rows = 0
        for game_dict in self.json_data:
            for move in game_dict["moves"]:
                rows += len(move.keys())  # 1 key means 1 move, 2 keys means 2 moves
        self.row_count = rows

    def generatre_matrix(self, save_data: bool = True) -> NDArray[np.float64]:
        self.process_raw_data()
        self._filter_by_moves(self.moves_to_skip)
        json_data = self.json_data
        rows = self.row_count
        output_final: NDArray[np.float64] = np.ndarray((rows, 71))
        i = 0
        for game in json_data:
            fen_list = self._create_rows(game)
            for j in fen_list:
                output_final[i] = j
                i += 1
                if i % self.verbose == 0 or i == self.row_count - 1:
                    logging.info("Processed %s rows", i)
        if save_data:
            output_file_path = (
                self.output_dir / f"{self.data_type}_d{self.sf_depth}.npy"
            )
            np.save(file=output_file_path, arr=output_final)
        return output_final

    def _filter_by_moves(self, moves_to_skip: int) -> None:
        """
        Allows for the removal of a certain number of moves at the start of the game
        in order to skip openings
        """
        data = self.json_data
        if moves_to_skip:
            # for ind, game in enumerate(data):
            # filtered_moves: list[MovesDict] = game["moves"][moves_to_skip:]
            # game["moves"] = filtered_moves
            # data[ind]["moves"] = game["moves"]
            for game in data:
                game.update({"moves": game["moves"][moves_to_skip:]})

        self.json_data = data
        self._count_rows()


def main(config_name: str) -> None:
    logging.debug("initialising processing...")
    data_dir = BASE_DIR / "data"
    raw_data_dir = data_dir / "raw"
    output_data_dir = data_dir / "processed"

    with open(BASE_DIR / "data_processing_config.yml") as config_file:
        config = safe_load(config_file)[config_name]

    data_type: str = config["data_type"]  # train / val / test
    raw_data_file_name: str = config["raw_data_file_name"]
    raw_data_path = raw_data_dir / data_type / raw_data_file_name
    processor = DataProcessing(
        raw_data_path=raw_data_path,
        output_dir=output_data_dir / data_type,
        sf_depth=config["sf_depth"],
        data_type=data_type,
    )
    _ = processor.generatre_matrix(save_data=False)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Config")
        parser.add_argument(
            "--config",
            type=str,
            default="windows",
            help="The name of the config file in the config.yml file",
        )

        args = parser.parse_args()
        config = args.config
    except SystemExit:
        config = "windows"
    main(config)


# %%

# %%
