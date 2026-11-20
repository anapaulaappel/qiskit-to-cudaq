"""
Pontuação de uma tradução com retorno em texto: a peça comum ao GEPA (métrica + feedback) e ao
GRPO (recompensa). Usa o mesmo verificador da avaliação (subprocesso isolado, referência fora dele)
e a mesma taxonomia de erros (eval/taxonomia.py, análise estática).

    from pontua import pontua
    r = pontua(tarefa, texto_do_modelo)          # tarefa: linha de data/*.jsonl ou de data/sft_*.jsonl
    r["score"], r["feedback"], r["status"], r["fidelidade"]

Modos de pontuação (`modo`):
  "binario"     1 se equivalente (fidelidade >= 0,999), senão 0 (o critério do artigo)
  "graduado"    1 se equivalente; 0,1 + 0,4·fidelidade se executa mas o estado difere; 0 se não
                executa (recompensa mais densa para o GRPO; a avaliação final continua binária)

A referência vem do campo "qiskit"/"qiskit_canonico" (tarefas de teste) ou do bloco Qiskit do
enunciado (pares de treino). É código do corpus, nunca do modelo.
"""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "verify"))
sys.path.insert(0, str(RAIZ / "eval"))

LIMIAR = 0.999
BLOCO = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

DICAS = {
    "sem_codigo_cudaq": "The answer has no CUDA-Q kernel. Write Python with `import cudaq` and a function "
                        "decorated with `@cudaq.kernel` named `kernel`.",
    "cuda_c": "This is CUDA C, not CUDA-Q. CUDA-Q kernels are Python functions decorated with `@cudaq.kernel`.",
    "api_qiskit": "Qiskit API calls (`qc.`, `QuantumCircuit`, `.append`) do not exist inside a CUDA-Q kernel. "
                  "Allocate `q = cudaq.qvector(n)` and call gates as functions, e.g. `h(q[0])`.",
    "idioma_qiskit": "Qiskit-only constructs such as `barrier()` or `measure_all()` have no CUDA-Q equivalent "
                     "here; drop them (the task asks for no measurement).",
    "classe_qiskit": "Qiskit gate classes (`XGate`, `CRY`, `Swap`) are not CUDA-Q gates. Use lowercase CUDA-Q "
                     "gates: `x`, `ry.ctrl(theta, control, target)`, `swap`.",
    "nome_porta_qiskit": "That is a Qiskit gate name. In CUDA-Q a controlled gate is written with `.ctrl`, "
                         "e.g. `cx(a, b)` -> `x.ctrl(q[a], q[b])`, `cz` -> `z.ctrl`, `cp(t)` -> `r1.ctrl(t, ...)`.",
    "porta_inventada": "That gate does not exist in CUDA-Q. Rotations are `rx`, `ry`, `rz` (not `r_y`), with the "
                       "angle first: `ry(theta, q[i])`.",
    "indice_cru": "Use qubits from a register, `q = cudaq.qvector(n)` and `q[i]`, not raw integers.",
    "qubits_errados": "The register size differs from the input circuit; allocate exactly as many qubits.",
    "kernel_argumentos": "The kernel must take no arguments.",
    "sintaxe": "The code is not valid Python.",
    "truncada": "The answer was cut off; write only the kernel, without explanations.",
    "porta_faltando": "The kernel runs but prepares a different state: some operations of the input are missing.",
    "porta_sobrando": "The kernel runs but prepares a different state: it has operations the input does not have.",
    "angulo_errado": "The kernel runs but prepares a different state: check rotation angles and their signs.",
    "controle_alvo": "The kernel runs but prepares a different state: control and target seem swapped; in "
                     "CUDA-Q `x.ctrl(control, target)` keeps Qiskit's `cx(control, target)` order.",
    "ordem_ou_qubit": "The kernel runs but prepares a different state: check which qubit each gate acts on.",
}


def _extrai(texto):
    blocos = BLOCO.findall(texto)
    return next((b for b in blocos if "cudaq" in b), blocos[0] if blocos else texto)


def _qiskit_da_tarefa(tarefa):
    if "qiskit_canonico" in tarefa or "qiskit" in tarefa:
        return tarefa.get("qiskit_canonico", tarefa.get("qiskit"))
    enunciado = tarefa["prompt"][0]["content"] if isinstance(tarefa["prompt"], list) else tarefa["prompt"]
    return BLOCO.findall(enunciado)[0]


def _cudaq_referencia(tarefa):
    if "cudaq" in tarefa:
        return tarefa["cudaq"]
    comp = tarefa.get("completion")
    if isinstance(comp, list):
        return _extrai(comp[0]["content"])
    return ""


def referencia(tarefa):
    from qiskit.quantum_info import Statevector
    ns = {}
    exec(_qiskit_da_tarefa(tarefa), ns)          # código do corpus, não do modelo
    from qiskit import QuantumCircuit
    qc = ns.get("qc")
    if not isinstance(qc, QuantumCircuit):   # estilos que renomeiam o circuito
        qc = next(v for v in ns.values() if isinstance(v, QuantumCircuit))
    return Statevector(qc.remove_final_measurements(inplace=False)).data   # estilos com medição


def pontua(tarefa, texto, modo="binario", ref=None, truncada=False):
    from equivalence import fidelidade
    from runner import executa
    import taxonomia
    ref = referencia(tarefa) if ref is None else ref
    res, psi = executa(_extrai(texto))
    status = res["status"]
    f = None
    if psi is not None:
        f = float(fidelidade(psi, ref))
        status = "ok"
    equivalente = f is not None and f >= LIMIAR
    if equivalente:
        return dict(score=1.0, fidelidade=f, status="ok", categoria=None,
                    feedback="Correct: the kernel prepares the same state as the Qiskit circuit.")
    n = len(ref).bit_length() - 1
    cat = taxonomia.classifica(texto, truncada, status, _cudaq_referencia(tarefa), n)
    erro = (res.get("erro") or "").strip().splitlines()
    detalhe = f" Verifier status: {status}" + (f" ({erro[-1][:160]})" if erro else "") + "."
    if f is not None:
        detalhe += f" State fidelity with the reference: {f:.3f} (needs >= {LIMIAR})."
    if modo == "graduado":
        score = 0.0 if f is None else 0.1 + 0.4 * f
    else:
        score = 0.0
    return dict(score=score, fidelidade=f, status=status, categoria=cat,
                feedback=DICAS.get(cat, "The translation is not equivalent.") + detalhe)


if __name__ == "__main__":                     # autoteste rápido com um par de treino
    import json
    par = json.loads(open(RAIZ / "data" / "sft_validacao.jsonl", encoding="utf8").readline())
    certo = par["completion"][0]["content"]
    errado = certo.replace("ry(", "r_y(").replace("rz(", "r_z(")
    qiskit_like = "```python\nimport cudaq\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(2)\n    barrier()\n```"
    for nome, txt in (("referência", certo), ("porta inventada", errado), ("idioma Qiskit", qiskit_like)):
        r = pontua(par, txt)
        print(f"{nome:16s} score={r['score']} cat={r['categoria']} | {r['feedback'][:150]}")
