import time
from collections import deque
from board import HexBoard
from MCTSplayer import SmartPlayer
from AlphaBeta_player import HeuristicPlayer

# Emojis para representar las fichas
EMPTY = "⚪"
PLAYER1 = "🔴"
PLAYER2 = "🔵"

def print_board(board: HexBoard, move=None, player=None):
    """
    Dibuja el tablero HEX con emojis:
    - 🔴 jugador 1 (rojo)
    - 🔵 jugador 2 (azul)
    - ⚪ vacío
    La última jugada se muestra sin resaltar (para mantener alineación).
    """
    size = board.size
    print("   " + "  ".join(f"{c:2}" for c in range(size)))
    for r in range(size):
        if r % 2 == 0:
            print(" ", end="")
        else:
            print("  ", end="")
        print(f"{r:2} ", end="")
        for c in range(size):
            cell = board.board[r][c]
            if cell == 0:
                symbol = EMPTY
            elif cell == 1:
                symbol = PLAYER1
            else:
                symbol = PLAYER2
            print(f" {symbol} ", end="")
        print()
    print(f"  {PLAYER1} = Jugador 1 (rojo)  |  {PLAYER2} = Jugador 2 (azul)  |  {EMPTY} = vacío")
    print("  Rojo conecta izquierda ↔ derecha; Azul conecta arriba ↔ abajo.\n")

def get_winning_path(board: HexBoard, player_id: int) -> list:
    """
    Devuelve el camino (lista de coordenadas) que conecta los bordes para el jugador,
    o None si no hay conexión.
    """
    size = board.size
    visited = [[False] * size for _ in range(size)]
    parent = [[None] * size for _ in range(size)]
    queue = deque()

    def neighbors(r, c):
        if r % 2 == 0:
            offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        else:
            offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                yield nr, nc

    if player_id == 1:
        for r in range(size):
            if board.board[r][0] == player_id:
                visited[r][0] = True
                queue.append((r, 0))
        while queue:
            r, c = queue.popleft()
            if c == size - 1:
                path = []
                while (r, c) != (None, None):
                    path.append((r, c))
                    r, c = parent[r][c] if parent[r][c] is not None else (None, None)
                return path[::-1]
            for nr, nc in neighbors(r, c):
                if not visited[nr][nc] and board.board[nr][nc] == player_id:
                    visited[nr][nc] = True
                    parent[nr][nc] = (r, c)
                    queue.append((nr, nc))
        return None
    else:  # player_id == 2
        for c in range(size):
            if board.board[0][c] == player_id:
                visited[0][c] = True
                queue.append((0, c))
        while queue:
            r, c = queue.popleft()
            if r == size - 1:
                path = []
                while (r, c) != (None, None):
                    path.append((r, c))
                    r, c = parent[r][c] if parent[r][c] is not None else (None, None)
                return path[::-1]
            for nr, nc in neighbors(r, c):
                if not visited[nr][nc] and board.board[nr][nc] == player_id:
                    visited[nr][nc] = True
                    parent[nr][nc] = (r, c)
                    queue.append((nr, nc))
        return None

def play_game(player1, player2, size=7, delay=0.3, verbose=True):
    board = HexBoard(size)
    current, other = player1, player2
    move_count = 0
    if verbose:
        print(f"\n--- {player1.__class__.__name__} (Rojo) vs {player2.__class__.__name__} (Azul) ---\n")
        print("Tablero inicial:")
        print_board(board)
        if delay > 0:
            time.sleep(delay)

    while True:
        move = current.play(board.clone())
        if move is None:
            print(f"¡Jugador {current.player_id} no devolvió movimiento! Pierde.")
            return other.player_id

        board.place_piece(move[0], move[1], current.player_id)
        move_count += 1

        if verbose:
            color_name = "Rojo" if current.player_id == 1 else "Azul"
            print(f"Jugada {move_count}: {color_name} coloca en {move}")
            print_board(board, move, current.player_id)
            if delay > 0:
                time.sleep(delay)

        if board.check_connection(current.player_id):
            if verbose:
                color_name = "Rojo" if current.player_id == 1 else "Azul"
                print(f"🎉 ¡{color_name} conecta en {move_count} movimientos! 🎉")
                path = get_winning_path(board, current.player_id)
                if path:
                    print(f"Camino de conexión: {' → '.join(str(p) for p in path)}")
                else:
                    print("¡ERROR! No se encontró camino pero check_connection dijo que sí.")
                print()
            return current.player_id

        current, other = other, current

def main():
    mcts = SmartPlayer(1)
    heuristic = HeuristicPlayer(2)

    tablero_size = 7
    num_partidas = 20
    mostrar_tablero = True
    pausa = 0.3

    wins = {1: 0, 2: 0}
    for i in range(num_partidas):
        if i % 2 == 0:
            print(f"\n{'='*50}\nPARTIDA {i+1}: MCTS (Rojo) empieza\n{'='*50}")
            winner = play_game(mcts, heuristic, size=tablero_size, delay=pausa, verbose=mostrar_tablero)
        else:
            print(f"\n{'='*50}\nPARTIDA {i+1}: Heurístico (Azul) empieza\n{'='*50}")
            winner = play_game(heuristic, mcts, size=tablero_size, delay=pausa, verbose=mostrar_tablero)
        wins[winner] += 1
        print(f"Resultado parcial: MCTS {wins[1]} - Heurístico {wins[2]}\n")
        if i < num_partidas - 1 and mostrar_tablero:
            time.sleep(1)

    print("\n" + "="*50)
    print("RESULTADO FINAL DESPUÉS DE 20 PARTIDAS")
    print("="*50)
    print(f"MCTS (Rojo)   : {wins[1]} victorias")
    print(f"Heurístico (Azul): {wins[2]} victorias")
    if wins[1] > wins[2]:
        print("🎉 ¡MCTS es el ganador! 🎉")
    elif wins[2] > wins[1]:
        print("🎉 ¡Heurístico es el ganador! 🎉")
    else:
        print("🤝 ¡Empate! 🤝")

if __name__ == "__main__":
    main()

