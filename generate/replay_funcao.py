"""Experimento B: replay no formato de função Qiskit (o formato dos QuantumKatas, sem usar os Katas).

O primeiro replay (CUDA-Q -> Qiskit, script solto) eliminou a mistura de sintaxe nos Katas, mas metade
das tarefas perdidas ficou sem `return qc`: o formato do replay vazou para a tarefa. Aqui o replay tem
o formato da tarefa que se quer proteger: um esqueleto de função que recebe `qc` e `qs`, com docstring
que descreve as portas, e a resposta é a função completa terminando em `return qc`.

Os exemplos vêm dos pares de TREINO das famílias sem laço (aleatório, GHZ, W); nada dos Katas. Cada
função é executada com qs = [0..n-1] e precisa dar o mesmo estado do circuito original.
Saída: data/replay_funcao.jsonl com {"id", "familia", "n", "prompt", "completion"}.
    python generate/replay_funcao.py
"""
import json, random, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "generate"))
from estilo import circuito, portas
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

SEMENTE, QUANTOS = 20261107, 339          # mesma quantidade do primeiro replay
MAX_PORTAS = 17
INSTRUCAO = "Complete the function below. Reply with the full Python code in one fenced block.\n\n"
NOME = {"h": "H", "x": "X", "y": "Y", "z": "Z", "s": "S", "t": "T", "rx": "RX", "ry": "RY", "rz": "RZ",
        "cx": "CNOT", "cz": "CZ", "swap": "SWAP", "cry": "controlled-RY"}


def descreve(g, ps, qs):
    a = [f"qs[{q}]" for q in qs]
    ang = f"({round(ps[0], 6)})" if ps else ""
    if g in ("cx", "cry"):
        return f"{NOME[g]}{ang} with control {a[0]} and target {a[1]}"
    if g in ("cz", "swap"):
        return f"{NOME[g]} on {a[0]} and {a[1]}"
    return f"{NOME[g]}{ang} on {a[0]}"


def chamada(g, ps, qs):
    args = [repr(round(p, 6)) for p in ps] + [f"qs[{q}]" for q in qs]
    return f"qc.{g}({', '.join(args)})"


def exemplo(t, r):
    gs = portas(circuito(t["qiskit"]))
    nome = r.choice(["apply_sequence", "prepare_state", "build_circuit", "apply_gates"])
    passos = "\n".join(f"        {i + 1}. {descreve(g, ps, qs)}" for i, (g, ps, qs) in enumerate(gs))
    cabecalho = (f"from qiskit import QuantumCircuit\n\ndef {nome}(qc: QuantumCircuit, qs: list) -> QuantumCircuit:\n"
                 f'    """\n    Apply the following gates, in order, to the qubits qs:\n{passos}\n\n'
                 f"    Args:\n        qc: Quantum circuit to modify\n        qs: List of {t['n']} qubit indices\n\n"
                 f'    Returns:\n        Modified quantum circuit\n    """\n')
    esqueleto = cabecalho + "    # Your code here\n    pass\n"
    corpo = "".join(f"    {chamada(g, ps, qs)}\n" for g, ps, qs in gs) + "    return qc\n"
    return INSTRUCAO + esqueleto, "```python\n" + cabecalho + corpo + "```", nome, cabecalho + corpo


def confere(t, nome, codigo):
    ns = {}
    exec(codigo, ns)                          # código gerado por nós
    qc = ns[nome](QuantumCircuit(t["n"]), list(range(t["n"])))
    ref, novo = Statevector(circuito(t["qiskit"])).data, Statevector(qc).data
    return abs(novo.conj() @ ref) ** 2 >= 1 - 1e-9


def main():
    r = random.Random(SEMENTE)
    validacao = {json.loads(l)["id"] for l in open(RAIZ / "data" / "sft_validacao.jsonl", encoding="utf8")}
    pares = [json.loads(l) for l in open(RAIZ / "data" / "treino.jsonl", encoding="utf8")]
    pares = [t for t in pares if t["id"] not in validacao and t["familia"] in ("aleatorio", "ghz", "w")]
    # A lista de portas aparece na docstring do enunciado e na resposta; com mais de 17 portas o exemplo
    # passa de max_length (1024 tokens) e a resposta seria truncada antes do `return qc`.
    pares = [t for t in pares if len(portas(circuito(t["qiskit"]))) <= MAX_PORTAS]
    saida = []
    for t in r.sample(pares, QUANTOS):
        prompt, completion, nome, codigo = exemplo(t, r)
        assert confere(t, nome, codigo), t["id"]
        saida.append({"id": "fun_" + t["id"], "familia": t["familia"], "n": t["n"], "prompt": prompt, "completion": completion})
    with open(RAIZ / "data" / "replay_funcao.jsonl", "w", encoding="utf8") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in saida)
    print(f"{len(saida)} exemplos de função verificados por execução")
    print(saida[0]["completion"][:700])


if __name__ == "__main__":
    main()
