import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from chess_eval.schemas import InputData
from stockfish import Stockfish  # type: ignore


def create_input(input_data: InputData) -> Tensor:
    total_time = 120
    turn_map = {"white": 0, "black": 1}

    sf = Stockfish(r"../stockfish/stockfish-windows-x86-64-avx2.exe", depth=20)
    fen = input_data.fen_number
    if not sf.is_fen_valid(fen):
        raise ValueError(f"Input must be a valid FEN - provided value is {fen}")

    sf.set_fen_position(fen)

    features = [
        convert_fen_to_matrix(fen),
        float(input_data.white_time) / total_time,
        float(input_data.black_time) / total_time,
        sf.get_evaluation()["value"] / 100,
        turn_map[input_data.turn.lower()],
        int(input_data.white_rating),
        int(input_data.black_rating),
    ]
    inner_array = features[0]
    remaining_elements = features[1:]
    X_array: NDArray[np.float32] = np.array(list(inner_array) + remaining_elements)

    with open("../models/scaling.json") as f:
        ratings_json = json.load(f)
    X_array[-2:] = (X_array[-2:] - ratings_json["min_rating"]) / (
        ratings_json["max_rating"] - ratings_json["min_rating"]
    )
    X = torch.tensor(X_array).to(torch.float32)
    logging.info(X)
    return X


def prep_data(data_path: Path, scaling_path: Path) -> tuple[Tensor, Tensor]:
    df = np.load(data_path)

    if "train" in data_path.root:
        ratings_json = {
            "min_rating": df[:, -3:-1].min(),
            "max_rating": df[:, -3:-1].max(),
        }
        with open(scaling_path, "w", encoding="utf-8") as f:
            json.dump(ratings_json, f)

    with open(scaling_path, encoding="utf-8") as f:
        ratings_json = json.load(f)

    df[:, -3:-1] = (df[:, -3:-1] - ratings_json["min_rating"]) / (
        ratings_json["max_rating"] - ratings_json["min_rating"]
    )
    np.random.shuffle(df)

    X = torch.tensor(df[:, :-1])
    y = torch.tensor(df[:, -1])

    y = F.one_hot(y.to(torch.int64))

    X = X.to(torch.float32)
    y = y.to(torch.float32)
    return X, y


def convert_time_str_to_seconds(time_str: str) -> int:
    """
    Input str format is [%emt 00:00:00]
    """
    time_match = re.search(r"\d+:\d+:\d+", time_str)
    if not time_match:
        raise ValueError(f"Time string incompatible - {time_str}")
    time = time_match.group()
    h, m, s = map(int, time.split(":"))
    return h * 3600 + m * 60 + s


def convert_fen_to_matrix(fen_string: str) -> NDArray[np.float64]:
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
                matrix[rank][file_index] = piece_values[char]  # type: ignore
                file_index += 1

    return np.array(matrix).flatten()
