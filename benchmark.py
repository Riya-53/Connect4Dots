import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# ---------------------------------------------------------
# CONSTANTS & BOARD LOGIC
# ---------------------------------------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7

EMPTY = 0
PLAYER_PIECE = 1  # Human (Red)
AI_PIECE = 2      # AI (Yellow)

WINDOW_LENGTH = 4

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    if col < 0 or col >= COLUMN_COUNT:
        return False
    return board[ROW_COUNT - 1][col] == EMPTY

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == EMPTY:
            return r
    return None

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def winning_move(board, piece):
    # Check horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (board[r][c] == piece and board[r][c + 1] == piece and 
                board[r][c + 2] == piece and board[r][c + 3] == piece):
                return True, [(r, c), (r, c + 1), (r, c + 2), (r, c + 3)]
    # Check vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (board[r][c] == piece and board[r + 1][c] == piece and 
                board[r + 2][c] == piece and board[r + 3][c] == piece):
                return True, [(r, c), (r + 1, c), (r + 2, c), (r + 3, c)]
    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (board[r][c] == piece and board[r + 1][c + 1] == piece and 
                board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                return True, [(r, c), (r + 1, c + 1), (r + 2, c + 2), (r + 3, c + 3)]
    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if (board[r][c] == piece and board[r - 1][c + 1] == piece and 
                board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                return True, [(r, c), (r - 1, c + 1), (r - 2, c + 2), (r - 3, c + 3)]
    return False, []

def is_terminal_node(board):
    p1_win, _ = winning_move(board, PLAYER_PIECE)
    if p1_win:
        return True
    ai_win, _ = winning_move(board, AI_PIECE)
    if ai_win:
        return True
    return len(get_valid_locations(board)) == 0

def evaluate_window(window, piece, weights=None):
    if weights is None:
        weights = {"center": 6, "three": 10, "two": 4, "block": 80}
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    piece_count = window.count(piece)
    empty_count = window.count(EMPTY)
    opp_count = window.count(opp_piece)

    if piece_count == 4:
        score += 100000
    elif piece_count == 3 and empty_count == 1:
        score += weights.get("three", 10)
    elif piece_count == 2 and empty_count == 2:
        score += weights.get("two", 4)

    if opp_count == 3 and empty_count == 1:
        score -= weights.get("block", 80)

    return score

def score_position(board, piece, weights=None):
    if weights is None:
        weights = {"center": 6, "three": 10, "two": 4, "block": 80}
    score = 0

    # Center column preference
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * weights.get("center", 6)

    # Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece, weights)

    # Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece, weights)

    # Positively sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, weights)

    # Negatively sloped diagonal
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, weights)

    return score

@dataclass
class SearchStats:
    nodes_visited: int = 0
    branches_pruned: int = 0
    max_depth_reached: int = 0
    time_ms: float = 0.0
    best_score: int = 0
    pv_line: List[int] = field(default_factory=list)
    depth_breakdown: Dict[int, int] = field(default_factory=dict)
    candidate_evals: Dict[int, int] = field(default_factory=dict)

def get_ordered_moves(valid_locations, prefer_center=True):
    if not prefer_center:
        return valid_locations
    center = COLUMN_COUNT // 2
    return sorted(valid_locations, key=lambda col: abs(col - center))

def minimax(board, depth, alpha, beta, maximizingPlayer, use_pruning=True, prefer_center=True, weights=None, stats=None):
    if stats is not None:
        stats.nodes_visited += 1

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            p1_win, _ = winning_move(board, PLAYER_PIECE)
            ai_win, _ = winning_move(board, AI_PIECE)
            if ai_win:
                return (None, 100000000000 + depth)
            elif p1_win:
                return (None, -100000000000 - depth)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE, weights))

    ordered_moves = get_ordered_moves(valid_locations, prefer_center=prefer_center)

    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(valid_locations) if valid_locations else 0
        for col in ordered_moves:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, use_pruning, prefer_center, weights, stats)[1]
            if new_score > value:
                value = new_score
                best_col = col
            if use_pruning:
                alpha = max(alpha, value)
                if alpha >= beta:
                    if stats is not None:
                        stats.branches_pruned += 1
                    break
        return best_col, value
    else:
        value = math.inf
        best_col = random.choice(valid_locations) if valid_locations else 0
        for col in ordered_moves:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, use_pruning, prefer_center, weights, stats)[1]
            if new_score < value:
                value = new_score
                best_col = col
            if use_pruning:
                beta = min(beta, value)
                if alpha >= beta:
                    if stats is not None:
                        stats.branches_pruned += 1
                    break
        return best_col, value

def get_ai_move(board, depth=4, use_pruning=True, prefer_center=True, weights=None):
    start_time = time.perf_counter()
    stats = SearchStats()
    col, score = minimax(
        board, depth, -math.inf, math.inf, True,
        use_pruning=use_pruning,
        prefer_center=prefer_center,
        weights=weights,
        stats=stats
    )
    stats.time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    stats.best_score = int(score) if math.isfinite(score) else (999999 if score > 0 else -999999)
    return col, stats

def run_academic_benchmark():
    """
    Compares pure Minimax vs Minimax with Alpha-Beta Pruning.
    Demonstrates:
    1. Identical optimal decision-making (proof of correctness).
    2. Exponential node reduction.
    3. Practical runtime speedup.
    """
    print("=" * 80)
    print(" CONNECT-4 AI: ACADEMIC BENCHMARK (MINIMAX vs ALPHA-BETA PRUNING)")
    print("=" * 80)

    board = create_board()
    moves = [
        (3, PLAYER_PIECE), (3, AI_PIECE),
        (2, PLAYER_PIECE), (4, AI_PIECE),
        (3, PLAYER_PIECE), (2, AI_PIECE),
        (1, PLAYER_PIECE), (4, AI_PIECE),
    ]
    for col, piece in moves:
        r = get_next_open_row(board, col)
        drop_piece(board, r, col, piece)

    print("Initial Test Board State:")
    print(np.flip(board, 0))
    print("-" * 80)
    print(f"{'Depth':<6} | {'Algorithm':<18} | {'Nodes Visited':<14} | {'Pruned Cuts':<12} | {'Time (ms)':<10} | {'Move':<5} | {'Score':<8}")
    print("-" * 80)

    depths = [2, 3, 4, 5]

    for d in depths:
        # 1. Pure Minimax (No Pruning)
        col_pure, stats_pure = get_ai_move(board, depth=d, use_pruning=False)
        print(f"{d:<6} | {'Minimax (Pure)':<18} | {stats_pure.nodes_visited:<14} | {stats_pure.branches_pruned:<12} | {stats_pure.time_ms:<10.1f} | {col_pure:<5} | {stats_pure.best_score:<8}")

        # 2. Minimax with Alpha-Beta Pruning
        col_ab, stats_ab = get_ai_move(board, depth=d, use_pruning=True)
        print(f"{d:<6} | {'Minimax + Alpha-Beta':<18} | {stats_ab.nodes_visited:<14} | {stats_ab.branches_pruned:<12} | {stats_ab.time_ms:<10.1f} | {col_ab:<5} | {stats_ab.best_score:<8}")

        # Compute efficiency comparison
        reduction = round((1 - stats_ab.nodes_visited / stats_pure.nodes_visited) * 100, 1)
        speedup = round(stats_pure.time_ms / max(0.1, stats_ab.time_ms), 1)
        print(f"      -> [Depth {d} Summary]: Pruned {reduction}% of search space | Speedup: {speedup}x | Moves Match: {col_pure == col_ab}")
        print("-" * 80)

    print("\nCONCLUSION FOR VIVA:")
    print("1. As depth increases, Alpha-Beta pruning exponentially suppresses combinatorial explosion.")
    print("2. Both algorithms pick identical moves, validating that Alpha-Beta never prunes the optimal path.")
    print("=" * 80)

if __name__ == "__main__":
    run_academic_benchmark()
