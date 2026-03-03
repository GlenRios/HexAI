import time
import math
from collections import deque
from player import Player
from board import HexBoard

class HeuristicPlayer(Player):
    """
    Jugador que utiliza búsqueda alfa-beta con profundidad iterativa
    y una heurística de distancia mínima (0-1 BFS) para evaluar posiciones.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 3 - player_id

    def play(self, board: HexBoard) -> tuple:
        start_time = time.time()
        time_limit = 4.5  # time

        moves = self._get_legal_moves(board)
        if not moves:
            return None

        for move in moves:
            test_board = board.clone()
            test_board.place_piece(move[0], move[1], self.player_id)
            if test_board.check_connection(self.player_id):
                return move

        for move in moves:
            test_board = board.clone()
            test_board.place_piece(move[0], move[1], self.opponent_id)
            if test_board.check_connection(self.opponent_id):
                return move

        best_move = None
        depth = 1
        while time.time() - start_time < time_limit:
            moves_sorted = sorted(moves, key=lambda m: self._heuristic_move(board, m, self.player_id), reverse=True)
            best_val = -math.inf
            alpha = -math.inf
            beta = math.inf

            for move in moves_sorted:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.player_id)
                val = self._alphabeta(new_board, depth - 1, alpha, beta, False, start_time, time_limit)
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
                if time.time() - start_time >= time_limit:
                    break

            depth += 1
            if time.time() - start_time >= time_limit:
                break

        return best_move if best_move is not None else moves[0]

    def _get_legal_moves(self, board):
        size = board.size
        return [(r, c) for r in range(size) for c in range(size) if board.board[r][c] == 0]

    def _heuristic_move(self, board, move, player_id):
        """Evaluación rápida de un movimiento (para ordenamiento)."""
        test_board = board.clone()
        test_board.place_piece(move[0], move[1], player_id)
        return self._evaluate(test_board, player_id)

    def _alphabeta(self, board, depth, alpha, beta, maximizing, start_time, time_limit):
        if time.time() - start_time >= time_limit:
            return self._evaluate(board, self.player_id if maximizing else self.opponent_id)

        if board.check_connection(self.player_id):
            return math.inf
        if board.check_connection(self.opponent_id):
            return -math.inf
        if depth == 0:
            return self._evaluate(board, self.player_id if maximizing else self.opponent_id)

        moves = self._get_legal_moves(board)
        if not moves:
            return 0 

        if maximizing:
            value = -math.inf
            moves_sorted = sorted(moves, key=lambda m: self._heuristic_move(board, m, self.player_id), reverse=True)
            for move in moves_sorted:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.player_id)
                val = self._alphabeta(new_board, depth - 1, alpha, beta, False, start_time, time_limit)
                value = max(value, val)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            moves_sorted = sorted(moves, key=lambda m: self._heuristic_move(board, m, self.opponent_id), reverse=True)
            for move in moves_sorted:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.opponent_id)
                val = self._alphabeta(new_board, depth - 1, alpha, beta, True, start_time, time_limit)
                value = min(value, val)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _evaluate(self, board, player_id):
        """Evalúa el tablero desde la perspectiva de player_id usando distancia mínima."""
        dist_self = self._distance_to_win(board, player_id)
        dist_opp = self._distance_to_win(board, 3 - player_id)

        if dist_self == 0:
            return math.inf
        if dist_opp == 0:
            return -math.inf
        if dist_self == float('inf'):
            return -math.inf
        if dist_opp == float('inf'):
            return math.inf

        return (dist_opp - dist_self) / (dist_self + dist_opp + 1)

    def _distance_to_win(self, board, player_id):
        """
        Calcula la distancia mínima (número de celdas vacías) que necesita player_id
        para conectar sus lados usando 0-1 BFS.
        Retorna float('inf') si es imposible.
        """
        size = board.size
        opponent = 3 - player_id

        if player_id == 1:  # conecta izquierda-derecha
            start_border = [(r, 0) for r in range(size)]
            target_border = [(r, size - 1) for r in range(size)]
        else:  # jugador 2 conecta arriba-abajo
            start_border = [(0, c) for c in range(size)]
            target_border = [(size - 1, c) for c in range(size)]

        dist = [[float('inf')] * size for _ in range(size)]
        dq = deque()

        for r, c in start_border:
            cell = board.board[r][c]
            if cell == opponent:
                continue
            if cell == player_id:
                dist[r][c] = 0
                dq.appendleft((r, c))
            else:  # vacía
                dist[r][c] = 1
                dq.append((r, c))

        def neighbors(r, c):
            if r % 2 == 0:  # fila par
                offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
            else:  # fila impar
                offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    yield nr, nc

        # 0-1 BFS
        while dq:
            r, c = dq.popleft()
            current = dist[r][c]
            for nr, nc in neighbors(r, c):
                cell = board.board[nr][nc]
                if cell == opponent:
                    continue
                cost = 0 if cell == player_id else 1
                if current + cost < dist[nr][nc]:
                    dist[nr][nc] = current + cost
                    if cost == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))

        # Buscar distancia mínima al borde objetivo
        min_dist = float('inf')
        for r, c in target_border:
            cell = board.board[r][c]
            if cell == opponent:
                continue
            if dist[r][c] < min_dist:
                min_dist = dist[r][c]
        return min_dist