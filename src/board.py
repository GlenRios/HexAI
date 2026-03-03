from collections import deque

class HexBoard:
    """
    Tablero de HEX con tamaño N x N y disposición even‑r (filas pares desplazadas a la derecha).
    - 0: vacío
    - 1: jugador 1 (rojo) – conecta izquierda ↔ derecha
    - 2: jugador 2 (azul) – conecta arriba ↔ abajo
    """
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def clone(self) -> 'HexBoard':
        """Devuelve una copia profunda del tablero."""
        new_board = HexBoard(self.size)
        for r in range(self.size):
            for c in range(self.size):
                new_board.board[r][c] = self.board[r][c]
        return new_board

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Coloca una ficha si la casilla está vacía."""
        if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0:
            self.board[row][col] = player_id
            return True
        return False

    def check_connection(self, player_id: int) -> bool:
        """
        Verifica si el jugador ha conectado sus dos lados mediante BFS.
        - Jugador 1: desde el borde izquierdo (col=0) hasta el derecho (col=size-1)
        - Jugador 2: desde el borde superior (row=0) hasta el inferior (row=size-1)
        """
        size = self.size
        visited = [[False] * size for _ in range(size)]
        queue = deque()

        def neighbors(r, c):
            if r % 2 == 0:  # fila par
                offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
            else:           # fila impar
                offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    yield nr, nc

        if player_id == 1:
            for r in range(size):
                if self.board[r][0] == player_id:
                    visited[r][0] = True
                    queue.append((r, 0))
            while queue:
                r, c = queue.popleft()
                if c == size - 1:
                    return True
                for nr, nc in neighbors(r, c):
                    if not visited[nr][nc] and self.board[nr][nc] == player_id:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            return False
        else: 
            for c in range(size):
                if self.board[0][c] == player_id:
                    visited[0][c] = True
                    queue.append((0, c))
            while queue:
                r, c = queue.popleft()
                if r == size - 1:
                    return True
                for nr, nc in neighbors(r, c):
                    if not visited[nr][nc] and self.board[nr][nc] == player_id:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            return False

    def __str__(self):
        """Representación simple del tablero (sin colores)."""
        return '\n'.join(' '.join(str(self.board[r][c]) for c in range(self.size)) for r in range(self.size))

