# WIP
import logging
from typing import Annotated

import chess
import chess.svg
import torch
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from chess_eval.constants import BASE_DIR, MODEL_PATH
from chess_eval.networks import Network
from chess_eval.schemas import InputData
from chess_eval.utils import create_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI()

# Allow local dev frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=BASE_DIR / "chess_eval" / "templates")

app.mount(
    "/static", StaticFiles(directory=BASE_DIR / "chess_eval" / "static"), name="static"
)


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "chess_eval" / "templates" / "index.html")


@app.post("/predict")
async def predict(request: Request, data: Annotated[InputData, Form()]) -> HTMLResponse:
    logging.critical(data)
    X = create_input(data)
    sf_eval = round(float(X[-4]), 2)
    logging.info("Stockfish Evaluation - %s", sf_eval)
    # Use the input in your machine learning model
    input_size = 70
    output_layer1 = 32
    output_layer2 = 16
    model = Network(
        input_size=input_size, output_layer1=output_layer1, output_layer2=output_layer2
    )

    model_state_dict = torch.load(MODEL_PATH)  # nosec: CWE-502

    model.load_state_dict(model_state_dict)

    X = X.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        print(y_pred)
        pred = y_pred.argmax().item()

    result_map = {"0": "White win", "1": "Draw", "2": "Black win"}
    result = result_map[str(pred)]
    # result_str = f"White win - {y_pred[0][0].item() * 100:.2f}%\nDraw - {y_pred[0][1].item() * 100:.2f}%\nBlack win - {y_pred[0][2].item() * 100:.2f}%"
    result_str = f"{'White win':<12} - {y_pred[0][0].item() * 100:>6.2f}%\n{'Draw':<12} - {y_pred[0][1].item() * 100:>6.2f}%\n{'Black win':<12} - {y_pred[0][2].item() * 100:>6.2f}%"

    print(result_str)

    board = chess.Board()
    board.set_fen(data.fen_number)
    image = chess.svg.board(board, size=400)
    # print(y_pred.tolist())
    list_y_pred = [round(i, 3) for i in y_pred.tolist()[0]]
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "result": result,
            "result_str": result_str,
            "svg_image": image,
            "sf_eval": sf_eval,
            "y_pred": list_y_pred,
        },
    )


# Lightweight JSON API for frontend consumption
_model_cached: Network | None = None


def _get_model() -> Network:
    global _model_cached
    if _model_cached is None:
        input_size = 70
        output_layer1 = 32
        output_layer2 = 16
        model = Network(
            input_size=input_size,
            output_layer1=output_layer1,
            output_layer2=output_layer2,
        )
        state = torch.load(MODEL_PATH)  # nosec: CWE-502
        model.load_state_dict(state)
        model.eval()
        _model_cached = model
    return _model_cached


@app.post("/api/predict")
async def api_predict(data: InputData) -> dict:
    X = create_input(data)
    sf_eval = round(float(X[-4]), 2)

    model = _get_model()
    with torch.no_grad():
        y_pred = model(X.unsqueeze(0))
        pred = int(y_pred.argmax().item())

    result_map = {0: "White win", 1: "Draw", 2: "Black win"}
    result = result_map[pred]
    probs = [float(round(v, 6)) for v in y_pred.squeeze(0).tolist()]
    return {
        "result": result,
        "probs": probs,
        "sf_eval": float(sf_eval),
        "fen": data.fen_number,
    }


# if __name__ == "__main__":
# uvicorn.run(app, host="localhost", port=8000)
