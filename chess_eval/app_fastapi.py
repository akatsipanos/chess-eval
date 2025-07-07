# WIP
import logging
from pathlib import Path
from typing import Annotated

import chess
import chess.svg
import torch
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from chess_eval.networks import Network
from chess_eval.schemas import InputData
from chess_eval.utils import create_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


app = FastAPI()

base_dir = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=base_dir / "templates")
app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse("index.html")


@app.post("/predict")
async def predict(request: Request, data: Annotated[InputData, Form()]) -> HTMLResponse:
    logging.critical(data)
    X = create_input(data)
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
    board.set_fen(data.fen_number)
    image = chess.svg.board(board, size=400)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "result": result,
            "result_str": result_str,
            "svg_image": image,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
