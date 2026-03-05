from player import Player
from board import HexBoard
import time
import math
import random
from collections import deque

class MixedPlayer(Player):
    """
    Jugador inteligente que combina dos estrategias:
    - Para tableros pequeños (N < 10): utiliza Monte Carlo Tree Search (MCTS) con simulaciones aleatorias.
    - Para tableros grandes (N >= 10): utiliza búsqueda Alpha-Beta con heurística de distancia mínima.
    Además, en ambos casos se comprueban jugadas ganadoras y amenazas del oponente.
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 3 - player_id

    # ------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------
    def play(self, board: HexBoard) -> tuple:
        # Comprobaciones rápidas comunes
        win_move = self._find_winning_move(board, self.player_id)
        if win_move:
            return win_move

        threat_move = self._find_winning_move(board, self.opponent_id)
        if threat_move:
            return threat_move

        # Decidir estrategia según el tamaño
        if board.size < 10:
            return self._mcts_play(board)
        else:
            return self._alphabeta_play(board)

    # ------------------------------------------------------------
    # MCTS para tableros pequeños
    # ------------------------------------------------------------
    class _MCTSNode:
        """Nodo para MCTS (versión simple, sin RAVE)."""
        def __init__(self, board, player_to_move, move, parent):
            self.board = board
            self.player_to_move = player_to_move
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.wins = 0
            self.player_made_move = None
            self.untried_moves = None  # se inicializará con movimientos legales

        def ucb_score(self, child, exploration=1.4):
            if child.visits == 0:
                return float('inf')
            return (child.wins / child.visits) + exploration * math.sqrt(math.log(self.visits) / child.visits)

        def select_child(self):
            return max(self.children, key=lambda c: self.ucb_score(c))

        def backpropagate(self, winner):
            self.visits += 1
            if self.player_made_move is not None and winner == self.player_made_move:
                self.wins += 1
            if self.parent:
                self.parent.backpropagate(winner)

    def _mcts_play(self, board: HexBoard) -> tuple:
        """Versión MCTS con simulaciones aleatorias y límite de tiempo."""
        start = time.time()
        time_limit = 4.5

        moves = self._get_legal_moves(board)
        if len(moves) == 1:
            return moves[0]

        root = self._MCTSNode(board, self.player_id, None, None)
        root.player_made_move = None
        root.untried_moves = moves[:]

        while time.time() - start < time_limit:
            node = root

            # Selección
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()

            # Expansión
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                new_board = node.board.clone()
                new_board.place_piece(move[0], move[1], node.player_to_move)
                next_player = 3 - node.player_to_move
                child = self._MCTSNode(new_board, next_player, move, node)
                child.player_made_move = node.player_to_move
                child.untried_moves = self._get_legal_moves(new_board)
                node.children.append(child)
                node = child

            # Simulación aleatoria
            winner = self._random_simulation(node.board, node.player_to_move)

            # Retropropagación
            node.backpropagate(winner)

        # Elegir el hijo con más visitas
        if not root.children:
            return moves[0]
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def _random_simulation(self, board, first_player):
        """Simulación completamente aleatoria hasta que alguien gane."""
        sim_board = board.clone()
        current = first_player
        while True:
            moves = self._get_legal_moves(sim_board)
            if not moves:
                break
            move = random.choice(moves)
            sim_board.place_piece(move[0], move[1], current)
            if sim_board.check_connection(current):
                return current
            current = 3 - current
        return None  # no debería ocurrir

    # ------------------------------------------------------------
    # Alpha-Beta para tableros grandes
    # ------------------------------------------------------------
    def _alphabeta_play(self, board: HexBoard) -> tuple:
        """Búsqueda con poda alfa-beta y heurística de distancia."""
        start_time = time.time()
        time_limit = 4.5

        moves = self._get_legal_moves(board)
        if not moves:
            return None

        # Profundidad dinámica: más profunda cuando quedan pocas celdas
        remaining = len(moves)
        max_depth = 5 if remaining < 10 else 3

        best_move = None
        depth = 1
        while time.time() - start_time < time_limit and depth <= max_depth:
            # Ordenar movimientos por heurística para mejor poda
            moves_sorted = sorted(moves, key=lambda m: self._heuristic_move_score(board, m, self.player_id), reverse=True)
            best_val = -math.inf
            alpha = -math.inf
            beta = math.inf

            for move in moves_sorted:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.player_id)
                val = self._alphabeta_rec(new_board, depth - 1, alpha, beta, False, start_time, time_limit)
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

    def _alphabeta_rec(self, board, depth, alpha, beta, maximizing, start_time, time_limit):
        """Llamada recursiva de alfa-beta."""
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
            moves_sorted = sorted(moves, key=lambda m: self._heuristic_move_score(board, m, self.player_id), reverse=True)
            for move in moves_sorted:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.player_id)
                val = self._alphabeta_rec(new_board, depth - 1, alpha, beta, False, start_time, time_limit)
                value = max(value, val)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            moves_sorted = sorted(moves, key=lambda m: self._heuristic_move_score(board, m, self.opponent_id), reverse=True)
            for move in moves_sorted:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.opponent_id)
                val = self._alphabeta_rec(new_board, depth - 1, alpha, beta, True, start_time, time_limit)
                value = min(value, val)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _evaluate(self, board, player_id):
        """Evalúa el tablero usando diferencia de distancias mínimas."""
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

    def _heuristic_move_score(self, board, move, player_id):
        """Puntuación rápida de un movimiento para ordenación (negativo de distancia)."""
        test_board = board.clone()
        test_board.place_piece(move[0], move[1], player_id)
        dist = self._distance_to_win(test_board, player_id)
        if dist == float('inf'):
            return -1e9
        return -dist

    # ------------------------------------------------------------
    # Utilidades comunes
    # ------------------------------------------------------------
    def _get_legal_moves(self, board):
        """Devuelve lista de celdas vacías."""
        size = board.size
        return [(r, c) for r in range(size) for c in range(size) if board.board[r][c] == 0]

    def _find_winning_move(self, board, player_id):
        """Busca una jugada que haga ganar a player_id inmediatamente."""
        size = board.size
        for r in range(size):
            for c in range(size):
                if board.board[r][c] == 0:
                    board.board[r][c] = player_id
                    if board.check_connection(player_id):
                        board.board[r][c] = 0
                        return (r, c)
                    board.board[r][c] = 0
        return None

    def _distance_to_win(self, board, player_id):
        """
        Calcula la distancia mínima (número de celdas vacías) que necesita player_id
        para conectar sus lados usando 0-1 BFS.
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

        min_dist = float('inf')
        for r, c in target_border:
            cell = board.board[r][c]
            if cell == opponent:
                continue
            if dist[r][c] < min_dist:
                min_dist = dist[r][c]
        return min_dist