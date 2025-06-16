import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from chess_eval.data_manipulation import convert_fen_to_matrix
from stockfish import Stockfish


def create_input(input_data: dict[str, Any]) -> Tensor:
    total_time = 120
    turn_map = {"white": 0, "black": 1}

    sf = Stockfish(r"/usr/local/Cellar/stockfish/15.1/bin/stockfish", depth=20)
    fen = input_data["fen_number"]
    if not sf.is_fen_valid():
        raise ValueError(f"Input must be a valid FEN - provided value is {fen}")

    sf.set_fen_position(fen)

    features = [
        convert_fen_to_matrix(fen),
        float(input_data["white_time"]) / total_time,
        float(input_data["black_time"]) / total_time,
        sf.get_evaluation()["value"] / 100,
        turn_map[input_data["turn"].lower()],
        int(input_data["white_rating"]),
        int(input_data["black_rating"]),
    ]
    inner_array = features[0]
    remaining_elements = features[1:]
    X = np.array(list(inner_array) + remaining_elements)

    with open("models/scaling.json") as f:
        ratings_json = json.load(f)
    X[-2:] = (X[-2:] - ratings_json["min_rating"]) / (
        ratings_json["max_rating"] - ratings_json["min_rating"]
    )
    X = torch.tensor(X).to(torch.float32)
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
