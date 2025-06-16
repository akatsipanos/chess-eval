import logging

import chess
import chess.svg
import torch
from flask import Flask, render_template, request

from chess_eval.networks import Network
from chess_eval.utils import create_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = Flask(__name__)


@app.route("/")
def home() -> str:
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    # Get user input from the forms
    input_data = {
        "fen_number": request.form["fen_number"],
        "white_time": request.form["white_time"],
        "black_time": request.form["black_time"],
        "white_rating": request.form["white_rating"],
        "black_rating": request.form["black_rating"],
        "turn": request.form["turn"],
    }

    X = create_input(input_data)

    # Use the input in your machine learning model
    input_size = 70
    output_layer1 = 32
    output_layer2 = 16
    model = Network(
        input_size=input_size, output_layer1=output_layer1, output_layer2=output_layer2
    )

    model_state_dict = torch.load("models/chess_model.pt")  # nosec: CWE-502

    model.load_state_dict(model_state_dict)

    X = X.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        print(y_pred)
        pred = y_pred.argmax().item()

    result_map = {"0": "White win", "1": "Draw", "2": "Black win"}
    result = result_map[str(pred)]
    result_str = f"White win - {y_pred[0][0].item() * 100:.2f}% Draw - {y_pred[0][1].item() * 100:.2f}% Black win - {y_pred[0][2].item() * 100:.2f}%"

    print(result_str)

    board = chess.Board()
    board.set_fen(input_data["fen_number"])
    image = chess.svg.board(board, size=400)

    # Return the prediction result to the user
    return render_template(
        "results.html", result=result, result_str=result_str, svg_image=image
    )


if __name__ == "__main__":
    # waitress.serve(app, port=5000, host='localhost')
    app.run()
