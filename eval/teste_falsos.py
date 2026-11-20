"""Prova o avaliador com modelos falsos, antes de gastar GPU. Sai com erro se algo não bater."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_eval import avalia, carrega_tarefas, extrai_codigo, wilson

T = carrega_tarefas()
VAZIO = "import cudaq\n\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector({n})\n"


def com_mz(c):
    return c.rstrip("\n") + "\n    mz(q)\n"


def outro_nome(c):
    return c.replace("def kernel():", "def meu_circuito():") + "\nprint(cudaq.sample(meu_circuito))\n"


MODELOS = {
    "perfeito": lambda t: t["completion"],
    "perfeito_conversador": lambda t: "Claro! Segue:\n\n" + t["completion"] + "\n\nEspero ajudar.",
    "perfeito_com_mz": lambda t: "```python\n" + com_mz(t["cudaq"]) + "```",
    "perfeito_outro_nome": lambda t: "```python\n" + outro_nome(t["cudaq"]) + "```",
    "lixo": lambda t: "Não sei fazer isso.",
    "devolve_qiskit": lambda t: "```python\n" + t["qiskit"] + "```",
    "valido_mas_errado": lambda t: "```python\n" + VAZIO.format(n=t["n"]) + "```",
    "qubits_a_menos": lambda t: "```python\n" + VAZIO.format(n=t["n"] - 1) + "```",
    "pede_argumento": lambda t: "```python\n" + t["cudaq"].replace("def kernel():", "def kernel(n: int):") + "```",
}
ESPERADO = {"perfeito": (1.0, 1.0), "perfeito_conversador": (1.0, 1.0), "perfeito_com_mz": (1.0, 1.0),
            "perfeito_outro_nome": (1.0, 1.0), "lixo": (0.0, 0.0), "devolve_qiskit": (0.0, 0.0),
            "pede_argumento": (0.0, 0.0)}
falhas = 0
for nome, f in MODELOS.items():
    g = avalia([{"id": i, "amostras": [f(t)]} for i, t in T.items()])["geral"]
    v, p = g["saida_valida"], g["pass_1"]
    print(f"{nome:22s} válida {v:.3f}  pass@1 {p:.3f}  falhas_ambiente {g['falhas_ambiente']}  {g['status']}")
    if nome in ESPERADO and (v, p) != ESPERADO[nome]:
        print("   ERRO: esperado", ESPERADO[nome]); falhas += 1
    if nome == "valido_mas_errado" and not (v == 1.0 and p < 0.05):
        print("   ERRO: validade e equivalência deveriam divergir"); falhas += 1
    if nome == "qubits_a_menos" and p != 0.0:
        print("   ERRO: número de qubits errado não pode passar"); falhas += 1

# pass@k: errar a primeira e acertar a segunda melhora pass@2, não pass@1.
t0 = next(iter(T.values()))
res = avalia([{"id": i, "amostras": ["lixo", t["completion"]]} for i, t in T.items()])["geral"]
print("pass@1 %.3f pass@2 %.3f" % (res["pass_1"], res["pass_2"]))
falhas += not (res["pass_1"] == 0.0 and res["pass_2"] == 1.0)
print("wilson(0,440) =", tuple(round(x, 4) for x in wilson(0, 440)), "wilson(220,440) =", tuple(round(x, 4) for x in wilson(220, 440)))
print("FALHOU" if falhas else "TUDO OK")
sys.exit(1 if falhas else 0)
