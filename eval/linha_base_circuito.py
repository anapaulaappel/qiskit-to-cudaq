"""Linha de base sem modelo, no nível do circuito (no espírito do Q-GEAR): EXECUTA o código Qiskit de
entrada, pega o QuantumCircuit resultante e traduz instrução por instrução para um kernel CUDA-Q.

Diferente de eval/linha_base_regras.py (que lê o texto), aqui o estilo de escrita não importa: o
Qiskit resolve registros, append(Gate()), funções auxiliares e numpy antes da tradução; o circuito é
transpilado (sem otimização) para as portas com regra, como faria um conversor de produção. barrier e
medidas são descartadas (o enunciado pede kernel sem medição). Executa o código de entrada do próprio
corpus (confiável), não código de modelo.
Saída: results/linha_base_circuito.json
"""
import json, math, sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
from qiskit import QuantumCircuit, transpile
from translate_eval import avalia, carrega_tarefas

UM = {"h": "h", "x": "x", "y": "y", "z": "z", "s": "s", "t": "t", "sdg": "sdg", "tdg": "tdg"}
ROT = {"rx": "rx", "ry": "ry", "rz": "rz", "p": "r1"}
CTRL1 = {"cx": "x", "cy": "y", "cz": "z", "ch": "h"}
CTRLROT = {"crx": "rx", "cry": "ry", "crz": "rz", "cp": "r1"}


def circuito(src):
    ns = {}
    exec(src, ns)
    qcs = [v for v in ns.values() if isinstance(v, QuantumCircuit)]
    return qcs[-1]


BASE = ["h", "x", "y", "z", "s", "t", "sdg", "tdg", "rx", "ry", "rz", "p", "cx", "cy", "cz", "ch",
        "crx", "cry", "crz", "cp", "swap", "ccx", "u"]


def traduz(src):
    # subcircuitos próprios e portas genéricas são reescritos nas portas com regra (sem otimizar)
    qc = transpile(circuito(src), basis_gates=BASE, optimization_level=0)
    idx = lambda b: qc.find_bit(b).index
    corpo = []
    for ins in qc.data:
        nome, qs, ps = ins.operation.name, [f"q[{idx(b)}]" for b in ins.qubits], ins.operation.params
        if nome in ("barrier", "measure"):
            continue
        if nome in UM: corpo.append(f"{UM[nome]}({qs[0]})")
        elif nome in ROT: corpo.append(f"{ROT[nome]}({float(ps[0])!r}, {qs[0]})")
        elif nome in CTRL1: corpo.append(f"{CTRL1[nome]}.ctrl({qs[0]}, {qs[1]})")
        elif nome in CTRLROT: corpo.append(f"{CTRLROT[nome]}.ctrl({float(ps[0])!r}, {qs[0]}, {qs[1]})")
        elif nome == "u": corpo.append(f"u3({float(ps[0])!r}, {float(ps[1])!r}, {float(ps[2])!r}, {qs[0]})")
        elif nome == "swap": corpo.append(f"swap({qs[0]}, {qs[1]})")
        elif nome in ("mcx", "ccx"): corpo.append(f"x.ctrl({', '.join(qs[:-1])}, {qs[-1]})")
        else: raise ValueError(f"porta sem regra: {nome}")
    linhas = ["import cudaq", "", "@cudaq.kernel", "def kernel():", f"    q = cudaq.qvector({qc.num_qubits})"]
    return "\n".join(linhas + ["    " + c for c in corpo]) + "\n"


T = carrega_tarefas()
saidas, falhas = [], {}
for i, t in T.items():
    try:
        saidas.append({"id": i, "amostras": ["```python\n" + traduz(t["qiskit"]) + "```"]})
    except Exception as e:
        falhas[i] = f"{type(e).__name__}: {e}"[:120]
        saidas.append({"id": i, "amostras": [""]})
res = avalia(saidas)
(RAIZ / "results" / "linha_base_circuito.json").write_text(json.dumps(
    {**{k: res[k] for k in ("limiar_fidelidade", "geral", "por_grupo")}, "falhas_traducao": falhas},
    indent=1, ensure_ascii=False))
print("geral", {k: res["geral"][k] for k in ("n", "saida_valida", "pass_1")}, "falhas de tradução", len(falhas))
for g, r in res["por_grupo"].items():
    if "/" not in g: print(f"{g:20s} n={r['n']:4d} pass@1={r['pass_1']:.3f}")
