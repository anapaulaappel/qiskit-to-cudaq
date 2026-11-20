"""Testa a lógica de regression.roda com tarefas escritas aqui, sem usar o dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression import roda
T = "def test_soma():\n    assert soma(2, 3) == 5\n"
C = [("passou", "def soma(a, b):\n    return a + b\n", T), ("falhou", "def soma(a, b):\n    return a - b\n", T),
     ("erro", "def outra():\n    pass\n", T), ("falhou", "def soma(a, b):\n    pass\n", T),
     ("sintaxe", "def (:\n", T), ("sem_teste", "x = 1\n", "y = 2\n"),
     ("timeout", "def soma(a, b):\n    while True:\n        pass\n", T)]
ruins = [(e, g) for e, c, t in C if (g := roda(c, t, timeout=3)) != e for e in [e]]
print("TUDO OK" if not ruins else ruins)
sys.exit(1 if ruins else 0)
