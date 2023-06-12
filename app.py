from flask import Flask, render_template, request
import torch
import torch.nn as nn

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = request.form['input_data']

    # Use the input in your machine learning model
    input_size = 70
    output_layer1= 32
    output_layer2 = 16
    model = nn.Sequential(nn.Linear(input_size,output_layer1),
                            nn.ReLU(),
                            nn.BatchNorm1d(num_features=output_layer1),
                            nn.Dropout(0.5),

                            nn.Linear(output_layer1,output_layer2),
                            nn.ReLU(),
                            nn.BatchNorm1d(num_features=output_layer2),
                            nn.Dropout(0.5),

                            nn.Linear(output_layer2,3),
                            nn.Softmax())
    model_state_dict = torch.load('chess_model.pt')

    model.load_state_dict(model_state_dict)
    result = model(input_data)

    # Return the prediction result to the user
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run()
