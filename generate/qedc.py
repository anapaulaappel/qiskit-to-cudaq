"""Conjunto fora da distribuição com código Qiskit real: kernels dos benchmarks QED-C
(SRI-International/QC-App-Oriented-Benchmarks, Apache-2.0, data/qedc_fonte/). O código entra quase
como está (sem a função kernel_draw, que só desenha), seguido de uma chamada que constrói o circuito
em tamanhos de 3 a 7 qubits. Caminhos sem medição no meio do circuito (method=1,
use_midcircuit_measurement=False, use_mcx_shim=False). A referência é o estado do circuito retornado,
sem as medições finais. Saída: data/teste_qedc.jsonl e data/tarefas_qedc.jsonl (formato do gerar_vllm).
"""
import ast, json, math, random, sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
F = RAIZ / "data" / "qedc_fonte"
sys.path.insert(0, str(RAIZ / "eval"))
from qiskit.quantum_info import Statevector

ENUNCIADO = ("Traduza este circuito Qiskit para um kernel CUDA-Q equivalente. Escreva um kernel chamado "
             "`kernel`, sem argumentos, que prepare o mesmo estado, sem medição.\n\n```python\n{c}\n```")


def fonte(arquivo):
    src = (F / arquivo).read_text()
    linhas = src.splitlines()
    corte = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "kernel_draw"]
    for n in sorted(corte, key=lambda n: -n.lineno):
        del linhas[n.lineno - 1:n.end_lineno]
    return "\n".join(linhas).rstrip() + "\n"


def bits(segredo, tam):
    return [int(b) for b in format(segredo, f"0{tam}b")[::-1]]


rng = random.Random(20261121)
casos = []
for n in (4, 5, 6, 7):
    s = rng.randrange(1, 2 ** (n - 1))
    casos.append(("bernstein_vazirani", "bv_kernel.py", n, f"qc = BersteinVazirani({n}, {s}, {bits(s, n - 1)}, method=1)"))
    casos.append(("deutsch_jozsa", "dj_kernel.py", n, f"qc = DeutschJozsa({n}, {n % 2})"))
    t = rng.randrange(1, 2 ** (n - 1)) / 2 ** (n - 1)
    casos.append(("phase_estimation", "pe_kernel.py", n, f"qc = PhaseEstimation({n}, {t}, False)"))
for n in (3, 4, 5, 6):
    s = rng.randrange(2 ** n)
    casos.append(("qft", "qft_kernel.py", n, f"qc = QuantumFourierTransform({n}, {s}, method=1)"))
    casos.append(("qft_method2", "qft_kernel.py", n, f"qc = QuantumFourierTransform({n}, {s}, method=2)"))
for n in (4, 6):
    for _ in range(2):
        s = rng.randrange(1, 2 ** n)
        casos.append(("hidden_shift", "hs_kernel.py", n, f"qc = HiddenShift({n}, {s}, {bits(s, n)}, method=1)"))
for n in (3, 4, 5):
    m = rng.randrange(2 ** n); it = int(math.pi / 4 * math.sqrt(2 ** n))
    casos.append(("grover", "grovers_kernel.py", n, f"qc = GroversSearch({n}, {m}, {it}, False)"))

teste, tarefas = [], []
for i, (fam, arq, n, chamada) in enumerate(casos):
    codigo = fonte(arq) + "\n" + chamada + "\n"
    canon = codigo + "qc = qc.remove_final_measurements(inplace=False)\n"
    ns = {}; exec(canon, ns)
    assert ns["qc"].num_qubits == n, (fam, n, ns["qc"].num_qubits)
    Statevector(ns["qc"])
    tid = f"qedc_{i:02d}_{fam}_{n}"
    teste.append({"id": tid, "familia": fam, "n": n, "qiskit": codigo, "qiskit_canonico": canon, "chamada": chamada})
    tarefas.append({"id": tid, "tipo": "traducao", "prompt": ENUNCIADO.format(c=codigo.rstrip())})
for nome, L in (("teste_qedc.jsonl", teste), ("tarefas_qedc.jsonl", tarefas)):
    (RAIZ / "data" / nome).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in L))
from collections import Counter
print(len(teste), "tarefas", dict(Counter(t["familia"] for t in teste)))
