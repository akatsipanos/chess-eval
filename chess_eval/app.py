from flask import Flask, render_template, request
import torch
from chess_eval.data_manipulation import convert_fen_to_matrix
from stockfish import Stockfish
import numpy as np
import json
import chess
import chess.svg
from chess_eval.networks import Network

def create_input(input_data):
    total_time = 120
    turn_map = {'white':0, 'black':1}

    sf  = Stockfish(r'/usr/local/Cellar/stockfish/15.1/bin/stockfish',depth=20)
    assert sf.is_fen_valid(input_data['fen_number'])
    sf.set_fen_position(input_data['fen_number'])

    features = [convert_fen_to_matrix(input_data['fen_number']),
                              float(input_data['white_time'])/total_time, 
                              float(input_data['black_time'])/total_time, 
                              sf.get_evaluation()['value']/100, 
                              turn_map[input_data['turn'].lower()],
                              int(input_data['white_rating']),
                              int(input_data['black_rating'])
                              ]
    inner_array = features[0]
    remaining_elements = features[1:]
    X = np.array(list(inner_array) + remaining_elements)

    with open('models/scaling.json', 'r') as f:
        ratings_json = json.load(f)
    X[-2:] = (X[-2:] - ratings_json['min_rating']) / (ratings_json['max_rating'] - ratings_json['min_rating'])
    X = torch.tensor(X).to(torch.float32)
    print(X)
    return X

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the forms
    input_data =  {'fen_number' : request.form['fen_number'],
                   'white_time' : request.form['white_time'],
                   'black_time' : request.form['black_time'],
                   'white_rating' : request.form['white_rating'],
                   'black_rating' : request.form['black_rating'],
                   'turn' : request.form['turn']
                   }
    
    X = create_input(input_data)

    # Use the input in your machine learning model
    input_size = 70
    output_layer1= 32
    output_layer2 = 16
    model = Network(input_size=input_size, output_layer1=output_layer1, output_layer2=output_layer2)
    
    model_state_dict = torch.load('models/chess_model.pt')

    model.load_state_dict(model_state_dict)

    X = X.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        print(y_pred)
        pred = y_pred.argmax().item()

    result_map = {'0':'White win','1':'Draw','2':'Black win'} 
    result = result_map[str(pred)]
    result_str =f'White win - {y_pred[0][0].item()*100:.2f}% Draw - {y_pred[0][1].item()*100:.2f}% Black win - {y_pred[0][2].item()*100:.2f}%'

    print(result_str)

    board = chess.Board()
    board.set_fen(input_data['fen_number'])
    image = chess.svg.board(board, size = 400)


    # Return the prediction result to the user
    return render_template('results.html', result=result, result_str=result_str, svg_image=image)

if __name__ == '__main__':
    # waitress.serve(app, port=5000, host='localhost')
    app.run()