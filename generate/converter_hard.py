"""Conjunto converter-hard: funções Qiskit PARAMETRIZADAS que devem virar um kernel CUDA-Q com os mesmos
parâmetros. Cada tarefa é verificada com três conjuntos de argumentos; um conversor que executa o Qiskit
com valores fixos produz um kernel sem parâmetros e não passa. Oito famílias (laço sobre n, ângulo
escalar, lista de ângulos, QFT, ramo sobre argumento booleano, codificação de inteiro em bits, rotações
controladas, funções auxiliares), seis variantes cada (portas, emaranhador, nomes, estilo de registro).
A referência CUDA-Q de cada tarefa é executada e conferida contra o Qiskit em todos os conjuntos.
Saída: data/teste_converter_hard.jsonl e data/tarefas_converter_hard.jsonl.
"""
import json, math, random, sys
from pathlib import Path
import numpy as np
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "verify"))
from qiskit.quantum_info import Statevector
from runner import executa_args
from equivalence import fidelidade

ENUNCIADO = ("Traduza esta função Qiskit para um kernel CUDA-Q equivalente. Escreva um kernel chamado `kernel` "
             "com os mesmos parâmetros, na mesma ordem, com anotações de tipo (`{sig}`), que prepare o mesmo "
             "estado que o circuito devolvido pela função, sem medição.\n\n```python\n{c}\n```")
rng = random.Random(20261122)
ROT = [("ry", "ry"), ("rx", "rx"), ("rz", "rz")]
ENT = [("cx", "x"), ("cz", "z")]


def cab(reg):
    return "from qiskit import QuantumCircuit, QuantumRegister\n" if reg else "from qiskit import QuantumCircuit\n"


def novo(nome, n, reg):
    return (f"    {nome} = QuantumRegister({n}, 'q')\n    qc = QuantumCircuit({nome})\n" if reg
            else f"    qc = QuantumCircuit({n})\n")


def idx(reg, nome, i):
    return f"{nome}[{i}]" if reg else i


def fam_ghz(v):
    fn = ["ghz", "make_ghz", "ghz_state", "build", "entangle_all", "cat_state"][v]; n = ["n", "num_qubits", "n", "width", "n", "size"][v]
    reg = v % 2 == 1; ent_q, ent_c = ENT[v % 2]; r = "qr"
    q = cab(reg) + f"\n\ndef {fn}({n}):\n" + novo(r, n, reg) + f"    qc.h({idx(reg, r, 0)})\n"
    q += f"    for i in range({n} - 1):\n        qc.{ent_q}({idx(reg, r, 'i')}, {idx(reg, r, 'i + 1')})\n" if ent_q == "cx" else \
         f"    for i in range({n} - 1):\n        qc.h({idx(reg, r, 'i + 1')})\n        qc.cz({idx(reg, r, 'i')}, {idx(reg, r, 'i + 1')})\n        qc.h({idx(reg, r, 'i + 1')})\n"
    q += "    return qc\n"
    c = f"@cudaq.kernel\ndef kernel({n}: int):\n    q = cudaq.qvector({n})\n    h(q[0])\n"
    c += f"    for i in range({n} - 1):\n        x.ctrl(q[i], q[i + 1])\n" if ent_q == "cx" else \
         f"    for i in range({n} - 1):\n        h(q[i + 1])\n        z.ctrl(q[i], q[i + 1])\n        h(q[i + 1])\n"
    return fn, q, c, f"kernel({n}: int)", [[3], [5], [6]]


def fam_ladder(v):
    rq, rc = ROT[v % 3]; eq, ec = ENT[v % 2]; fn = ["ladder", "rotation_chain", "prep", "ansatz", "layer", "chain"][v]
    reg = v >= 3; r = "q"
    q = cab(reg) + f"\n\ndef {fn}(n, theta):\n" + novo(r, "n", reg)
    if rq == "rz": q += f"    for i in range(n):\n        qc.h({idx(reg, r, 'i')})\n"
    q += f"    for i in range(n):\n        qc.{rq}(theta * (i + 1), {idx(reg, r, 'i')})\n"
    q += f"    for i in range(n - 1):\n        qc.{eq}({idx(reg, r, 'i')}, {idx(reg, r, 'i + 1')})\n    return qc\n"
    c = "@cudaq.kernel\ndef kernel(n: int, theta: float):\n    q = cudaq.qvector(n)\n"
    if rq == "rz": c += "    h(q)\n"
    c += f"    for i in range(n):\n        {rc}(theta * (i + 1), q[i])\n"
    c += f"    for i in range(n - 1):\n        {ec}.ctrl(q[i], q[i + 1])\n"
    return fn, q, c, "kernel(n: int, theta: float)", [[3, 0.3], [4, 1.1], [6, -0.7]]


def fam_layers(v):
    rq, rc = ROT[(v + 1) % 3] if v % 3 != 1 else ROT[0]; eq, ec = ENT[v % 2]; fn = ["hea", "layered", "variational", "circuit", "build_ansatz", "hw_efficient"][v]
    q = cab(False) + f"\n\ndef {fn}(n, layers, params):\n    qc = QuantumCircuit(n)\n    k = 0\n    for _ in range(layers):\n"
    q += f"        for i in range(n):\n            qc.{rq}(params[k], i)\n            k += 1\n"
    q += f"        for i in range(n - 1):\n            qc.{eq}(i, i + 1)\n    return qc\n"
    c = "@cudaq.kernel\ndef kernel(n: int, layers: int, params: list[float]):\n    q = cudaq.qvector(n)\n"
    c += f"    for l in range(layers):\n        for i in range(n):\n            {rc}(params[l * n + i], q[i])\n"
    c += f"        for i in range(n - 1):\n            {ec}.ctrl(q[i], q[i + 1])\n"
    a = lambda n, L: [n, L, [round(rng.uniform(-3, 3), 4) for _ in range(n * L)]]
    return fn, q, c, "kernel(n: int, layers: int, params: list[float])", [a(3, 1), a(4, 2), a(5, 2)]


def fam_qft(v):
    fn = ["qft", "fourier", "qft_circuit", "make_qft", "transform", "qft_n"][v]; sw = v % 2 == 0; prep = v >= 3
    q = "import math\n" + cab(False) + f"\n\ndef {fn}(n):\n    qc = QuantumCircuit(n)\n"
    if prep: q += "    for i in range(0, n, 2):\n        qc.x(i)\n"
    q += "    for j in reversed(range(n)):\n        qc.h(j)\n        for k in reversed(range(j)):\n            qc.cp(math.pi / 2 ** (j - k), k, j)\n"
    if sw: q += "    for i in range(n // 2):\n        qc.swap(i, n - 1 - i)\n"
    q += "    return qc\n"
    c = "import math\n\n\n@cudaq.kernel\ndef kernel(n: int):\n    q = cudaq.qvector(n)\n"
    if prep: c += "    for i in range(0, n, 2):\n        x(q[i])\n"
    c += "    for j in range(n - 1, -1, -1):\n        h(q[j])\n        for k in range(j - 1, -1, -1):\n            r1.ctrl(math.pi / 2 ** (j - k), q[k], q[j])\n"
    if sw: c += "    for i in range(n // 2):\n        swap(q[i], q[n - 1 - i])\n"
    return fn, q, c, "kernel(n: int)", [[2], [3], [5]]


def fam_branch(v):
    fn = ["prepare", "init_state", "choose", "make_state", "variant", "select_prep"][v]; flag = ["flip", "use_x", "excited", "mode", "invert", "alt"][v]
    rq, rc = ROT[v % 3]
    q = cab(False) + f"\n\ndef {fn}(n, {flag}, theta):\n    qc = QuantumCircuit(n)\n    if {flag}:\n        for i in range(n):\n            qc.x(i)\n"
    q += f"        qc.{rq}(theta, 0)\n    else:\n        qc.h(0)\n        for i in range(1, n):\n            qc.cx(0, i)\n    return qc\n"
    c = f"@cudaq.kernel\ndef kernel(n: int, {flag}: bool, theta: float):\n    q = cudaq.qvector(n)\n    if {flag}:\n        x(q)\n        {rc}(theta, q[0])\n"
    c += "    else:\n        h(q[0])\n        for i in range(1, n):\n            x.ctrl(q[0], q[i])\n"
    return fn, q, c, f"kernel(n: int, {flag}: bool, theta: float)", [[3, True, 0.4], [4, False, 0.4], [5, True, -1.2]]


def fam_bits(v):
    fn = ["encode", "basis_state", "load_int", "prepare_value", "bits", "to_basis"][v]; val = ["value", "mask", "k", "number", "data", "idx"][v]
    tail = v % 3
    q = cab(False) + f"\n\ndef {fn}(n, {val}):\n    qc = QuantumCircuit(n)\n    for i in range(n):\n        if ({val} >> i) & 1:\n            qc.x(i)\n"
    if tail == 1: q += "    qc.h(n - 1)\n"
    if tail == 2: q += "    for i in range(n - 1):\n        qc.cx(i, i + 1)\n"
    q += "    return qc\n"
    c = f"@cudaq.kernel\ndef kernel(n: int, {val}: int):\n    q = cudaq.qvector(n)\n    for i in range(n):\n        if ({val} >> i) & 1:\n            x(q[i])\n"
    if tail == 1: c += "    h(q[n - 1])\n"
    if tail == 2: c += "    for i in range(n - 1):\n        x.ctrl(q[i], q[i + 1])\n"
    return fn, q, c, f"kernel(n: int, {val}: int)", [[3, 5], [4, 9], [6, 38]]


def fam_ctrlrot(v):
    rq, rc = [("cry", "ry"), ("crx", "rx"), ("crz", "rz")][v % 3]; fn = ["cascade", "controlled_chain", "crot", "propagate", "cr_ladder", "spread"][v]
    q = cab(False) + f"\n\ndef {fn}(n, angles):\n    qc = QuantumCircuit(n)\n    qc.h(0)\n    for i in range(n - 1):\n        qc.{rq}(angles[i], i, i + 1)\n"
    if v >= 3: q += "    qc.x(n - 1)\n"
    q += "    return qc\n"
    c = f"@cudaq.kernel\ndef kernel(n: int, angles: list[float]):\n    q = cudaq.qvector(n)\n    h(q[0])\n    for i in range(n - 1):\n        {rc}.ctrl(angles[i], q[i], q[i + 1])\n"
    if v >= 3: c += "    x(q[n - 1])\n"
    a = lambda n: [n, [round(rng.uniform(0.2, 3), 4) for _ in range(n - 1)]]
    return fn, q, c, "kernel(n: int, angles: list[float])", [a(3), a(4), a(6)]


def fam_helpers(v):
    rq, rc = ROT[v % 3]; eq, ec = ENT[(v // 3) % 2]; fn = ["circuit", "build", "model", "program", "make", "assemble"][v]
    q = cab(False) + f"\n\ndef rotations(qc, theta):\n    for i in range(qc.num_qubits):\n        qc.{rq}(theta, i)\n\n\n"
    q += f"def entangle(qc):\n    for i in range(qc.num_qubits - 1):\n        qc.{eq}(i, i + 1)\n\n\n"
    q += f"def {fn}(n, theta, reps):\n    qc = QuantumCircuit(n)\n    for _ in range(reps):\n        rotations(qc, theta)\n        entangle(qc)\n    return qc\n"
    c = f"@cudaq.kernel\ndef kernel(n: int, theta: float, reps: int):\n    q = cudaq.qvector(n)\n    for r in range(reps):\n"
    c += f"        for i in range(n):\n            {rc}(theta, q[i])\n        for i in range(n - 1):\n            {ec}.ctrl(q[i], q[i + 1])\n"
    return fn, q, c, "kernel(n: int, theta: float, reps: int)", [[3, 0.5, 1], [4, 1.3, 2], [5, -0.8, 3]]


FAMILIAS = {"ghz_n": fam_ghz, "ladder": fam_ladder, "layers": fam_layers, "qft_n": fam_qft, "branch": fam_branch,
            "bits": fam_bits, "ctrl_rot": fam_ctrlrot, "helpers": fam_helpers}


def estado_qiskit(src, fn, args):
    ns = {}; exec(src, ns)
    return Statevector(ns[fn](*args)).data


teste, tarefas, ruins = [], [], []
for fam, f in FAMILIAS.items():
    for v in range(6):
        fn, q, c, sig, argsets = f(v)
        ref_c = "import cudaq\n" + ("" if c.startswith("import") else "\n\n") + c
        res, estados = executa_args(ref_c, argsets)
        fids = [fidelidade(e, estado_qiskit(q, fn, a)) for e, a in zip(estados, argsets)] if estados else []
        if not estados or min(fids) < 0.999:
            ruins.append((fam, v, res, fids)); continue
        tid = f"hard_{fam}_{v}"
        teste.append({"id": tid, "familia": fam, "n": max(a[0] for a in argsets), "qiskit": q, "funcao": fn,
                      "assinatura": sig, "args": argsets, "cudaq": ref_c})
        tarefas.append({"id": tid, "tipo": "traducao", "prompt": ENUNCIADO.format(sig=sig, c=q.rstrip())})
for nome, L in (("teste_converter_hard.jsonl", teste), ("tarefas_converter_hard.jsonl", tarefas)):
    (RAIZ / "data" / nome).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in L))
print(len(teste), "tarefas válidas;", len(ruins), "descartadas", ruins[:4])
