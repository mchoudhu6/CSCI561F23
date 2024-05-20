import random
def read_input_file(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Extract player, times, and board from the lines
    player = lines[0].strip()
    my_time, opponent_time = map(float, lines[1].split())

    # Extract the board from the remaining lines
    board = []
    for line in lines[2:]:
        row = line.strip()
        board.append(list(row))  # Convert the row string to a list of characters

    return player, my_time, opponent_time, board

def convert_to_alphabetical(column_index):
    """Convert column index to alphabetical representation."""
    return chr(column_index + ord('a'))

def write_output_file(chosen_move, output_file):
    """Write the chosen move to the output file."""
    with open(output_file, 'w') as f:
        f.write(convert_to_alphabetical(chosen_move[1]-1) + str(chosen_move[0]))

def is_valid_move(board, row, col, color):
    """Check if there are any neighbors of the opposite color."""
    opposite = 'X' if color == 'O' else 'O'
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    valid_moves = []

    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 1 <= r <= 12 and 1 <= c <= 12 and board[r - 1][c - 1] == opposite:
            # Keep moving in the same direction until an empty space or player's piece is encountered
            r += dr
            c += dc
            while (1 <= r <= 12 and 1 <= c <= 12):
                if board[r - 1][c - 1] == color:
                    # Another player's piece found, stop searching in this direction
                    break
                elif board[r - 1][c - 1] == '.':
                    # Empty space found, place the piece and return the position
                    valid_moves.append((r, c))
                    break
                r += dr
                c += dc
    for dr, dc in valid_moves:
        r, c = row + dr, col + dc
        while 1 <= r <= 12 and 1 <= c <= 12:
            r += dr
            c += dc
    # If no valid move found, return None
    return valid_moves if valid_moves else []
    

def evaluate(board, player):
    opponent = 'O' if player == 'X' else 'X'
    
    # Evaluate mobility
    player_mobility = sum(1 for row in range(len(board)) for col in range(len(board[row])) if is_valid_move(board, row, col, player))
    opponent_mobility = sum(1 for row in range(len(board)) for col in range(len(board[row])) if is_valid_move(board, row, col, opponent))
    mobility_score = player_mobility - opponent_mobility
    
    # Evaluate coin parity
    player_coins = sum(row.count(player) for row in board)
    opponent_coins = sum(row.count(opponent) for row in board)
    coin_parity_score = player_coins - opponent_coins
    
    # Evaluate corner and edge control
    corner_weight = 25
    edge_weight = 10
    corner_control = sum([board[i][j] == player for i in [0, len(board)-1] for j in [0, len(board)-1]])
    edge_control = sum([board[i][j] == player for i in [0, len(board)-1] for j in range(2, len(board)-2)]) + \
                   sum([board[i][j] == player for i in range(2, len(board)-2) for j in [0, len(board)-1]])
    corner_edge_score = corner_weight * corner_control + edge_weight * edge_control
    
    # Evaluate stability - count stable discs
    # Implement stability calculation based on your algorithm
    stability_score = calculate_stability(board, player)
    
    # Evaluate potential mobility
    # Implement potential mobility calculation based on your algorithm
    player_potential_mobility = sum(potential_mobility(board, row, col, player) for row in range(len(board)) for col in range(len(board[row])))
    opponent_potential_mobility = sum(potential_mobility(board, row, col, opponent) for row in range(len(board)) for col in range(len(board[row])))
    potential_mobility_score = player_potential_mobility - opponent_potential_mobility
    
    
    
    # Combine individual scores with appropriate weights
    total_score = (mobility_score * 0.1) + (coin_parity_score * 0.2) + (corner_edge_score * 0.3) + (stability_score * 0.2) + (potential_mobility_score * 0.2)
    
    return total_score

def potential_mobility(board, row, col, color):
    # Get the opponent's color
    opponent_color = 'O' if color == 'X' else 'X'
    
    # Initialize the count for potential mobility
    potential_mobility_count = 0
    
    # Get the list of valid moves for the opponent
    valid_moves = is_valid_move(board, row, col, opponent_color)
    
    # Count the number of potential mobility moves for the opponent
    if valid_moves:
        potential_mobility_count = len(valid_moves)
    
    return potential_mobility_count

def calculate_stability(board, player):
    # Define the opponent's color
    opponent = 'O' if player == 'X' else 'X'

    # Initialize stability scores for both players
    player_stability = 0
    opponent_stability = 0

    # Iterate through each cell in the board
    for row in range(len(board)):
        for col in range(len(board[row])):
            # Check if the current cell contains the player's disc
            if board[row][col] == player:
                # Check if the disc is stable by ensuring it's not in a vulnerable position
                if is_stable_disc(board, row, col, player):
                    player_stability += 1
            # Check if the current cell contains the opponent's disc
            elif board[row][col] == opponent:
                # Check if the opponent's disc is stable
                if is_stable_disc(board, row, col, opponent):
                    opponent_stability += 1

    # Return the difference in stability scores between the player and the opponent
    return player_stability - opponent_stability


def is_stable_disc(board, row, col, color):
    # Define the opponent's color
    opponent = 'O' if color == 'X' else 'X'

    # Check if the disc is in the corner or on the edge
    if (row, col) in [(0, 0), (0, len(board)-1), (len(board)-1, 0), (len(board)-1, len(board)-1)]:
        return True
    elif row == 0 or col == 0 or row == len(board)-1 or col == len(board)-1:
        return True

    # Check stability in the row
    row_stable = all(board[row][j] == color for j in range(len(board[row])))
    if row_stable:
        return True

    # Check stability in the column
    col_stable = all(board[i][col] == color for i in range(len(board)))
    if col_stable:
        return True

    # Check stability in the main diagonal
    if row == col:
        diag_stable = all(board[i][i] == color for i in range(len(board)))
        if diag_stable:
            return True

    # Check stability in the anti-diagonal
    if row + col == len(board) - 1:
        anti_diag_stable = all(board[i][len(board)-1-i] == color for i in range(len(board)))
        if anti_diag_stable:
            return True

    # If none of the stability conditions are met, the disc is not stable
    return False
   
def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player, player, row, col):
    if depth == 0 or not any('.' in row for row in board):
        return evaluate(board, player)

    valid_moves = is_valid_move(board, row, col, player)

    if maximizing_player:
        max_eval = float('-inf')
        for move in valid_moves:
            new_board = apply_move(board, move[0], move[1], player)
            eval = minimax_alpha_beta(new_board, depth - 1, alpha, beta, False, player, move[0], move[1])
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_board = apply_move(board, move[0], move[1], player)
            eval = minimax_alpha_beta(new_board, depth - 1, alpha, beta, True, player, move[0], move[1])
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def apply_move(board, row, col, player):
    new_board = [row[:] for row in board]  # Create a copy of the board
    new_board[row - 1][col - 1] = player  # Place the player's piece at the specified row and column
    return new_board

    
def main(): 
            
    input_file = 'input.txt' 
    output_file = 'output.txt'
            
    # Read input from the file
    player, my_time, opponent_time, board = read_input_file(input_file)
    print("Player:", player)
    print("My time:", my_time)
    print("Opponent's time:", opponent_time)
    print("Board:")
    for row in board:
        print(row)

    # Find all player piece positions on the board
    player_piece_positions = []
    for row_idx, row in enumerate(board):
        for col_idx, cell in enumerate(row):
            if cell == player:
                player_piece_positions.append((row_idx + 1, col_idx + 1))

    valid_moves = []  # Initialize a list to store all valid moves
    
    # Iterate through all player pieces
    for row, col in player_piece_positions:
        print(f"Checking piece at: {convert_to_alphabetical(col - 1)}{row}")
        piece_valid_moves = is_valid_move(board, row, col, player)
        valid_moves.extend(piece_valid_moves)  # Add valid moves for this piece to the list
        
    print("All Valid Moves:")
    for move in valid_moves:
        print(f"Row: {move[0]}, Column: {move[1]}")

    # Initialize variables to keep track of the best move and its score
    best_move = None
    best_score = float('-inf')

    # Evaluate each valid move using minimax with alpha-beta pruning
    for move in valid_moves:
        # Simulate the move and evaluate its score using minimax with alpha-beta pruning
        new_board = [row[:] for row in board]  # Create a copy of the board
        new_row, new_col = move
        score = minimax_alpha_beta(new_board, 4, float('-inf'), float('inf'), False, player, new_row, new_col)  # Evaluate the move
        # Update the best move and score if the current move has a higher score
        if score > best_score:
            best_score = score
            best_move = move

    # Print the chosen move
    if best_move:
        print("Chosen move: ", (convert_to_alphabetical(best_move[1]-1), best_move[0]))

        # Print the updated board (You might need to update the board with the chosen move)
        print("Updated Board after placing player piece:")
        updated_board = apply_move(board, best_move[0], best_move[1], player)
        for row in updated_board:
            print(row)

        # Write the output to the file
        write_output_file(best_move, output_file)
    else: 
        print("No valid moves found for", input_file)

if __name__ == "__main__":
    main()

        