import time
from collections import deque
from board import HexBoard

class TimeoutException(Exception):
    """Excepción personalizada para cuando un jugador excede el tiempo límite."""
    pass

def get_winning_path(board: HexBoard, player_id: int) -> list:
    """Devuelve el camino de conexión (lista de coordenadas) o None."""
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
    else:
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

def get_move_with_timeout(player, board, timeout=5.0):
    """
    Obtiene un movimiento del jugador con control de tiempo.
    
    Args:
        player: El jugador
        board: Tablero actual
        timeout: Tiempo máximo en segundos
    
    Returns:
        tuple: (movimiento, tiempo_tomado)
    
    Raises:
        TimeoutException: Si el jugador excede el tiempo límite
    """
    start_time = time.time()
    move = player.play(board.clone())
    elapsed = time.time() - start_time
    
    if elapsed > timeout:
        raise TimeoutException(f"Jugador {player.player_id} excedió el tiempo límite de {timeout}s (tomó {elapsed:.2f}s)")
    
    return move, elapsed

def play_single_game(player1, player2, size=7, time_limit=5.0, verbose=False):
    """
    Juega una partida entre dos jugadores con control de tiempo.
    
    Args:
        player1: Primer jugador (con player_id=1)
        player2: Segundo jugador (con player_id=2)
        size: Tamaño del tablero
        time_limit: Tiempo máximo por movimiento en segundos
        verbose: Si es True, imprime información de la partida
    
    Returns:
        tuple: (ganador, movimientos_totales, tiempo_total, descalificado)
        donde descalificado es el player_id del jugador descalificado (0 si none)
    """
    board = HexBoard(size)
    current, other = player1, player2
    move_count = 0
    game_start = time.time()
    
    if verbose:
        print(f"\n--- Nueva partida: {player1.__class__.__name__} ({player1.player_id}) vs {player2.__class__.__name__} ({player2.player_id}) ---")
    
    while True:
        try:
            # Obtener movimiento del jugador actual con control de tiempo
            move, move_time = get_move_with_timeout(current, board, time_limit)
            
            if move is None:
                if verbose:
                    print(f"¡Jugador {current.player_id} no devolvió movimiento! Descalificado.")
                return other.player_id, move_count, time.time() - game_start, current.player_id
            
            # Verificar que el movimiento sea válido (dentro del tablero y casilla vacía)
            row, col = move
            if not (0 <= row < size and 0 <= col < size and board.board[row][col] == 0):
                if verbose:
                    print(f"¡Jugador {current.player_id} hizo un movimiento inválido {move}! Descalificado.")
                return other.player_id, move_count, time.time() - game_start, current.player_id
            
            board.place_piece(row, col, current.player_id)
            move_count += 1
            
            if verbose and move_time > time_limit * 0.8:  # Advertencia si se acerca al límite
                print(f"⚠️  Jugador {current.player_id} tardó {move_time:.2f}s")
            
            # Verificar victoria
            if board.check_connection(current.player_id):
                elapsed = time.time() - game_start
                if verbose:
                    print(f"✓ Jugador {current.player_id} gana en {move_count} movimientos ({elapsed:.2f}s)")
                return current.player_id, move_count, elapsed, 0
            
            # Cambiar turno
            current, other = other, current
            
        except TimeoutException as e:
            if verbose:
                print(f"⏰ {e}")
            return other.player_id, move_count, time.time() - game_start, current.player_id

def tournament(player1, player2, num_games=10, size=7, time_limit=5.0, verbose=True):
    """
    Enfrenta a dos jugadores en múltiples partidas con control de tiempo.
    
    Args:
        player1: Primer jugador (debe tener player_id=1)
        player2: Segundo jugador (debe tener player_id=2)
        num_games: Número de partidas a jugar
        size: Tamaño del tablero
        time_limit: Tiempo máximo por movimiento en segundos
        verbose: Mostrar información detallada
    
    Returns:
        dict: Estadísticas del torneo
    """
    # Validar IDs de jugadores
    if player1.player_id != 1 or player2.player_id != 2:
        raise ValueError("Los jugadores deben tener player_id=1 y player_id=2 respectivamente")
    
    results = {
        'player1_wins': 0,
        'player2_wins': 0,
        'player1_disqualified': 0,
        'player2_disqualified': 0,
        'player1_moves': [],
        'player2_moves': [],
        'player1_times': [],
        'player2_times': [],
        'total_moves': [],
        'total_times': [],
        'disqualifications': []
    }
    
    print(f"\n{'='*70}")
    print(f"TORNEO: {player1.__class__.__name__} vs {player2.__class__.__name__}")
    print(f"{num_games} partidas - Tablero {size}x{size} - Tiempo límite: {time_limit}s por movimiento")
    print(f"{'='*70}")
    
    for game in range(1, num_games + 1):
        print(f"\n--- Partida {game}/{num_games} ---")
        
        # Alternar quién empieza
        if game % 2 == 1:
            winner, moves, elapsed, disq = play_single_game(player1, player2, size, time_limit, verbose)
            
            if disq == 1:
                results['player1_disqualified'] += 1
                results['player2_wins'] += 1  # Victoria por descalificación
                results['disqualifications'].append(f"Partida {game}: {player1.__class__.__name__} descalificado")
            elif disq == 2:
                results['player2_disqualified'] += 1
                results['player1_wins'] += 1
                results['disqualifications'].append(f"Partida {game}: {player2.__class__.__name__} descalificado")
            elif winner == 1:
                results['player1_wins'] += 1
                results['player1_moves'].append(moves)
                results['player1_times'].append(elapsed)
            else:
                results['player2_wins'] += 1
                results['player2_moves'].append(moves)
                results['player2_times'].append(elapsed)
        else:
            winner, moves, elapsed, disq = play_single_game(player2, player1, size, time_limit, verbose)
            
            if disq == 2:
                results['player2_disqualified'] += 1
                results['player1_wins'] += 1
                results['disqualifications'].append(f"Partida {game}: {player2.__class__.__name__} descalificado")
            elif disq == 1:
                results['player1_disqualified'] += 1
                results['player2_wins'] += 1
                results['disqualifications'].append(f"Partida {game}: {player1.__class__.__name__} descalificado")
            elif winner == 2:
                results['player2_wins'] += 1
                results['player2_moves'].append(moves)
                results['player2_times'].append(elapsed)
            else:
                results['player1_wins'] += 1
                results['player1_moves'].append(moves)
                results['player1_times'].append(elapsed)
        
        results['total_moves'].append(moves)
        results['total_times'].append(elapsed)
        
        # Mostrar marcador parcial
        print(f"\n📊 Marcador parcial ({game}/{num_games}):")
        print(f"  {player1.__class__.__name__}: {results['player1_wins']} victorias ({results['player1_disqualified']} descalificaciones)")
        print(f"  {player2.__class__.__name__}: {results['player2_wins']} victorias ({results['player2_disqualified']} descalificaciones)")
        
        if game < num_games and verbose:
            print("-" * 50)
    
    return results

def print_tournament_results(results, player1_name, player2_name):
    """Muestra los resultados del torneo de forma clara."""
    total_games = results['player1_wins'] + results['player2_wins']
    total_disqualifications = results['player1_disqualified'] + results['player2_disqualified']
    
    print(f"\n{'='*70}")
    print("RESULTADOS FINALES DEL TORNEO")
    print(f"{'='*70}")
    print(f"\nPartidas jugadas: {total_games}")
    print(f"Descalificaciones totales: {total_disqualifications}")
    
    if results['disqualifications']:
        print("\n📋 Historial de descalificaciones:")
        for dq in results['disqualifications']:
            print(f"  • {dq}")
    
    print(f"\n{player1_name}:")
    print(f"  Victorias: {results['player1_wins']} ({results['player1_wins']/total_games*100:.1f}%)")
    print(f"  Descalificaciones: {results['player1_disqualified']}")
    if results['player1_moves']:
        print(f"  Movimientos promedio (victorias): {sum(results['player1_moves'])/len(results['player1_moves']):.1f}")
        print(f"  Tiempo promedio (victorias): {sum(results['player1_times'])/len(results['player1_times']):.2f}s")
    
    print(f"\n{player2_name}:")
    print(f"  Victorias: {results['player2_wins']} ({results['player2_wins']/total_games*100:.1f}%)")
    print(f"  Descalificaciones: {results['player2_disqualified']}")
    if results['player2_moves']:
        print(f"  Movimientos promedio (victorias): {sum(results['player2_moves'])/len(results['player2_moves']):.1f}")
        print(f"  Tiempo promedio (victorias): {sum(results['player2_times'])/len(results['player2_times']):.2f}s")
    
    if results['total_moves']:
        print(f"\n📊 Estadísticas globales:")
        print(f"  Movimientos promedio por partida: {sum(results['total_moves'])/len(results['total_moves']):.1f}")
        print(f"  Tiempo promedio por partida: {sum(results['total_times'])/len(results['total_times']):.2f}s")
    
    # Determinar ganador (considerando descalificaciones)
    if results['player1_wins'] > results['player2_wins']:
        winner_emoji = "🏆" if results['player1_disqualified'] == 0 else "⚠️"
        print(f"\n{winner_emoji} ¡{player1_name} es el GANADOR! {winner_emoji}")
    elif results['player2_wins'] > results['player1_wins']:
        winner_emoji = "🏆" if results['player2_disqualified'] == 0 else "⚠️"
        print(f"\n{winner_emoji} ¡{player2_name} es el GANADOR! {winner_emoji}")
    else:
        print(f"\n🤝 ¡EMPATE! 🤝")
    
    print(f"{'='*70}\n")

# Ejemplo de uso
if __name__ == "__main__":
    # Importar tus jugadores
    from AlphaBeta_player import HeuristicPlayer
    from Mixed_player import MixedPlayer
    
    # Crear jugadores
    jugador2 = HeuristicPlayer(2)
    jugador1 = MixedPlayer(1)
    
    # Configuración del torneo
    NUM_GAMES = 10
    BOARD_SIZE = 7
    TIME_LIMIT = 5.0  # segundos por movimiento
    VERBOSE = True
    
    print(f"\n{'#'*70}")
    print(f"# TESTER DE HEX - CON VERIFICACIÓN DE TIEMPO")
    print(f"{'#'*70}")
    print(f"\nJugador 1: {jugador1.__class__.__name__}")
    print(f"Jugador 2: {jugador2.__class__.__name__}")
    print(f"\nConfiguración:")
    print(f"  • Partidas: {NUM_GAMES}")
    print(f"  • Tablero: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"  • Tiempo límite: {TIME_LIMIT} segundos por movimiento")
    print(f"  • Modo: {'Detallado' if VERBOSE else 'Resumido'}")
    
    input("\nPresiona Enter para comenzar el torneo...")
    
    # Ejecutar torneo
    resultados = tournament(
        jugador1, 
        jugador2, 
        num_games=NUM_GAMES, 
        size=BOARD_SIZE, 
        time_limit=TIME_LIMIT,
        verbose=VERBOSE
    )
    
    # Mostrar resultados
    print_tournament_results(resultados, 
                            jugador1.__class__.__name__, 
                            jugador2.__class__.__name__)