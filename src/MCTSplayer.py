from player import Player
from board import HexBoard
import random
import time
import math

class SmartPlayer(Player):
    """
    Jugador autónomo para HEX basado en Monte Carlo Tree Search (MCTS).
    Decide la mejor jugada usando UCT dentro de un límite de tiempo.
    """

    class Node:
        """Nodo del árbol MCTS."""
        def __init__(self, board, player_to_move, move, parent):
            self.board = board               # estado del tablero
            self.player_to_move = player_to_move  # jugador que debe mover en este nodo
            self.move = move                  # movimiento que llevó a este nodo (None para raíz)
            self.parent = parent
            self.children = []
            self.visits = 0
            self.wins = 0                      # victorias para el jugador que realizó el movimiento
            self.player_made_move = None       # jugador que hizo el movimiento (se asigna al crear hijo)
            self.untried_moves = None           # lista de movimientos no explorados

        def ucb_score(self, child, exploration=1.4):
            """Calcula el índice UCB para un hijo."""
            if child.visits == 0:
                return float('inf')
            return (child.wins / child.visits) + exploration * math.sqrt(math.log(self.visits) / child.visits)

        def select_child(self):
            """Selecciona el hijo con mayor UCB."""
            return max(self.children, key=lambda c: self.ucb_score(c))

        def backpropagate(self, winner):
            """Propaga el resultado de la simulación hacia arriba."""
            self.visits += 1
            if self.player_made_move is not None and winner == self.player_made_move:
                self.wins += 1
            if self.parent:
                self.parent.backpropagate(winner)

    def play(self, board: HexBoard) -> tuple:
        """
        Método principal que debe implementar la jugada.
        Recibe una copia del tablero actual y devuelve (fila, columna).
        """
        
        winning_move = self._find_winning_move(board, self.player_id)
        if winning_move:
            return winning_move
        start_time = time.time()
        time_limit = 4.5  # time
        root_player = self.player_id

        root = self.Node(board, root_player, move=None, parent=None)
        root.player_made_move = None
        root.untried_moves = self._get_legal_moves(board)

        if len(root.untried_moves) == 1:
            return root.untried_moves[0]

        while time.time() - start_time < time_limit:
            node = root
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()

            winner = self._check_terminal(node.board)
            if winner is not None:
                node.backpropagate(winner)
                continue

            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                new_board = node.board.clone()
                new_board.place_piece(move[0], move[1], node.player_to_move)
                next_player = 3 - node.player_to_move  # oponente

                child = self.Node(new_board, next_player, move, node)
                child.player_made_move = node.player_to_move
                child.untried_moves = self._get_legal_moves(new_board)
                node.children.append(child)

                sim_winner = self._simulate(child.board, child.player_to_move)

                child.backpropagate(sim_winner)

        if not root.children:
            return random.choice(self._get_legal_moves(board))

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    # ------------------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------------------
    def _get_legal_moves(self, board: HexBoard):
        """Devuelve lista de celdas vacías (fila, columna)."""
        moves = []
        size = board.size
        for r in range(size):
            for c in range(size):
                if board.board[r][c] == 0:
                    moves.append((r, c))
        return moves

    def _check_terminal(self, board: HexBoard):
        """Retorna el id del jugador ganador si lo hay, None en caso contrario."""
        if board.check_connection(1):
            return 1
        if board.check_connection(2):
            return 2
        return None
    
    def _find_winning_move(self, board: HexBoard, player_id: int):
        """Retorna la primera celda que al colocar una ficha hace que player_id gane."""
        size = board.size
        for r in range(size):
            for c in range(size):
                if board.board[r][c] == 0:
                    # Simular colocación
                    board.board[r][c] = player_id
                    if board.check_connection(player_id):
                        board.board[r][c] = 0  # restaurar
                        return (r, c)
                    board.board[r][c] = 0
        return None

    def _simulate(self, board: HexBoard, first_player):
        """
        Simula una partida aleatoria desde el estado dado.
        Devuelve el id del jugador ganador.
        """
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

        if sim_board.check_connection(1):
            return 1
        if sim_board.check_connection(2):
            return 2
        return None