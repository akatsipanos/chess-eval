#%%
import chess
import re
from stockfish import Stockfish
import numpy as np

def split_games(path):

    with open(path, "r") as file:
        data = file.readlines()

    games = []
    current_game = []

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

def processing(path = 'games/train/2018_03_11-2023_05_25.pgn'):
    games = split_games(path)
    games = [x for x in games ][:1000]
    ratings = [[int(x[5][-6:-2]), int(x[6][-6:-2])] for x in games]
    moves_games = [x[-3] for x in games]
    moves = [re.split('\d\.|\d\d\.',x[:x.find('} {')+2])[1:] for x in moves_games]
    result_map = {'1-0':0,'1/2':1,'0-1':2} # 0 white win, 1 draw, 2 black win
    results = [result_map[x[-3:]] for x in moves_games]
    game_times = 120*60
    return ratings, moves, results, game_times

def create_rows(move_list, total_time, ratings, results,depth):
    board = chess.Board()
    output = []
    wrt,brt = total_time, total_time
    sf  = Stockfish(r'/usr/local/Cellar/stockfish/15.1/bin/stockfish',depth=depth)

    for move_time in move_list:
        times = re.findall(r"\{[^}]+\}", move_time)
        time_strings = [x[-10:-2] for x in times]

        moves = (move_time.split(" ")[1], move_time.split(" ")[-4]) if len(move_time)>25 else (move_time.split(" ")[1],0)

        for i,move in enumerate(moves):
            if move == 0:
                continue
            else:
                    
                if i==0:
                    wrt-=time_to_seconds(time_strings[i])
                    turn = 0
                elif i ==1:
                    brt-=time_to_seconds(time_strings[i])
                    turn = 1

                fen = board.fen() # Get the fen for the position
                sf.set_fen_position(fen) # For stockfish evaluation

                features = [convert_fen_to_matrix(fen),
                              wrt/total_time, 
                              brt/total_time, 
                              sf.get_evaluation()['value']/100, 
                              turn,
                              ratings[0],
                              ratings[1],
                              results]
                inner_array = features[0]
                remaining_elements = features[1:]
                final_array = np.array(list(inner_array) + remaining_elements)
                output.append(final_array)
                

                board.push_san(move)
    return output

def time_to_seconds(time_str):
    hours, minutes, seconds = map(int, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

def convert_fen_to_matrix(fen_string):
    piece_values = {
        'P': 0.1, 'N': 0.3, 'B': 0.35, 'R': 0.5, 'Q': 0.9, 'K': 1.0,
        'p': -0.1, 'n': -0.3, 'b': -0.35, 'r': -0.5, 'q': -0.9, 'k': -1.0
    }

    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(8)] for _ in range(8)]

    # Split FEN string into relevant parts
    fen_parts = fen_string.split()
    board_part = fen_parts[0]
    ranks = board_part.split('/')

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

def count_rows(moves):
    rows = 0
    for move in moves:
        for i in move:
            if len(i) > 25:
                rows+=2
            else:
                rows+=1
    return rows

def create_final_matrix(rows, moves,ratings,game_times,results,depth,verbose):

    output_final = np.ndarray((rows,71))
    i = 0
    for moves_in_each_game, rating,result in zip(moves,ratings,results):
        fen_list = create_rows(moves_in_each_game,game_times, rating, result,depth)
        for j in fen_list:
            output_final[i] = j
            i+=1
            if i%verbose==0 or i==rows-1:
                print(i)
    return output_final

def save_data(path,data):
    np.save(file = path, arr = data)
    # print('[INFO] dataset successfully saved')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-o','--output-path', type = str, required=True, help='output dataset path')
    ap.add_argument('-r','--raw-data-path', type = str, required=True, help='path to raw data')
    ap.add_argument('-v', '--verbose', default=500, type=int, help='number of rows processed before print statement')
    ap.add_argument('-d', '--stockfish-depth', default=10, type=int, help='depth of stockfish engine evaluation')
    args = ap.parse_args()

    print('[INFO] initial processing...')
    ratings, moves, results, game_times = processing(args.raw_data_path)
    print(f'[INFO] total number of games: {len(moves)}')
    rows = count_rows(moves)
    print(f'[INFO] total number of rows: {rows}')
    print('[INFO] creating final dataset...')
    output_final = create_final_matrix(rows,moves,ratings,game_times, results,args.stockfish_depth, args.verbose,)

    save_data(args.output_path, output_final)
        
    return output_final


if __name__ == '__main__':
    output = main()

# Need to sort out the output. Final output should be an array of shape (452215,70)



# %%
