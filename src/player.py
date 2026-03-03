from board import HexBoard
class Player:
    """Clase base para todos los jugadores."""
    def __init__(self, player_id: int):
        self.player_id = player_id  # 1 o 2

    def play(self, board: 'HexBoard') -> tuple:
        """Debe ser implementado por la subclase."""
        raise NotImplementedError("Implementa este método")