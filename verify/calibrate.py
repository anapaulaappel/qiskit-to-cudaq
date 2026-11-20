"""Q0: calibra get_state, ordenação de qubits e convenção das portas. Roda só na CPU."""
import importlib.util, math, random, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cudaq
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from equivalence import fidelidade, inverte_bits

cudaq.set_target("qpp-cpu")

@cudaq.kernel
def x0():
    q = cudaq.qvector(3)
    x(q[0])

@cudaq.kernel
def ghz_com_mz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    mz(q)

@cudaq.kernel
def ghz_sem_mz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])

s = cudaq.get_state(x0)
a = np.array(s)
print("get_state:", type(s).__name__, "| np.array:", a.dtype, a.shape)
print("X em q0, CUDA-Q indice", int(np.argmax(abs(a))))
qc = QuantumCircuit(3); qc.x(0)
b = Statevector(qc).data
print("X em q0, Qiskit  indice", int(np.argmax(abs(b))))
print("fid direta %.6f | fid com inversão %.6f" % (fidelidade(a, b), fidelidade(inverte_bits(a), b)))

for nome, k in (("sem mz", ghz_sem_mz), ("com mz", ghz_com_mz)):
    try:
        print("GHZ(4)", nome, np.round(np.array(cudaq.get_state(k, 4)), 3)[[0, 15]], "norma %.6f" % np.linalg.norm(np.array(cudaq.get_state(k, 4))))
    except Exception as e:
        print("GHZ(4)", nome, "ERRO", type(e).__name__, str(e)[:160])
g = QuantumCircuit(4); g.h(0); [g.cx(i, i + 1) for i in range(3)]
print("GHZ(4) fid %.9f" % fidelidade(np.array(cudaq.get_state(ghz_sem_mz, 4)), Statevector(g).data))

# As 12 portas, circuitos aleatórios assimétricos: mapeamento 1 para 1, sem biblioteca de conversão.
UM = ["h", "x", "y", "z", "s", "t"]; ANG = ["rx", "ry", "rz"]
def aleatorio(n, d, r):
    qc = QuantumCircuit(n); linhas = [f"def k():", f"    q = cudaq.qvector({n})"]
    for _ in range(d):
        p = r.choice(UM + ANG + ["cx", "cz", "swap"])
        if p in UM:
            a = r.randrange(n); getattr(qc, p)(a); linhas.append(f"    {p}(q[{a}])")
        elif p in ANG:
            a, th = r.randrange(n), r.uniform(-math.pi, math.pi)
            getattr(qc, p)(th, a); linhas.append(f"    {p}({th!r}, q[{a}])")
        else:
            a, b = r.sample(range(n), 2)
            getattr(qc, p)(a, b)
            linhas.append({"cx": f"    x.ctrl(q[{a}], q[{b}])", "cz": f"    z.ctrl(q[{a}], q[{b}])",
                           "swap": f"    swap(q[{a}], q[{b}])"}[p])
    return qc, "@cudaq.kernel\n" + "\n".join(linhas) + "\n"

r = random.Random(7); fids, dir_ = [], []
for i in range(60):
    n = r.randint(2, 7); qc, src = aleatorio(n, r.randint(5, 40), r)
    # @cudaq.kernel lê o fonte com inspect: exec() falha, o kernel precisa estar em arquivo.
    with tempfile.TemporaryDirectory() as tmp:
        arq = Path(tmp) / f"cand_{i}.py"
        arq.write_text("import cudaq\n" + src)
        spec = importlib.util.spec_from_file_location(arq.stem, arq)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        psi = np.array(cudaq.get_state(mod.k))
    ref = Statevector(qc).data
    fids.append(fidelidade(inverte_bits(psi), ref)); dir_.append(fidelidade(psi, ref))
print("60 aleatórios, n=2..7: fid direta min %.12f | com inversão media %.3f" % (min(dir_), np.mean(fids)))

# mz no kernel: colapso? aleatório?
vistos = {}
for _ in range(40):
    v = np.array(cudaq.get_state(ghz_com_mz, 4))
    k = int(np.argmax(abs(v))); vistos[k] = vistos.get(k, 0) + 1
print("get_state com mz, 40 repetições, base colapsada:", vistos)

# Ordem das strings de sample vs ordem do vetor de estado.
print("sample X em q0, CUDA-Q:", dict(cudaq.sample(x0, shots_count=20).items()))
from qiskit_aer import AerSimulator
m3 = QuantumCircuit(3); m3.x(0); m3.measure_all()
print("sample X em q0, Qiskit:", AerSimulator().run(m3, shots=20).result().get_counts())

# A métrica enxerga fase relativa? Mesmas contagens, estados diferentes.
p = QuantumCircuit(1); p.h(0)
m = QuantumCircuit(1); m.h(0); m.z(0)
print("|+> vs |->: fid %.6f (contagens iguais)" % fidelidade(Statevector(p).data, Statevector(m).data))
