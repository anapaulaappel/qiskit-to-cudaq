"""Q1: gera os pares e só mantém os que batem por execução. Saída em data/*.jsonl."""
import argparse, concurrent.futures as cf, hashlib, json, random, sys, time
from collections import Counter
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "generate"), str(RAIZ / "verify")]
from families import FAMILIAS
from equivalence import fidelidade
from runner import executa
from qiskit.quantum_info import Statevector

SEMENTE, LIMIAR = 20261104, 0.999
# (divisão, família, tamanhos, quantidade pedida)
PLANO = [("treino", "aleatorio", [2, 3, 4, 5], 1100), ("treino", "ghz", [2, 3, 4, 5], 100),
         ("treino", "w", [2, 3, 4, 5], 100), ("treino", "qft", [2, 3, 4], 100),
         ("treino", "grover", [2, 3, 4, 5], 150),
         ("teste_tamanho", "aleatorio", [6, 7], 120), ("teste_tamanho", "ghz", [6, 7], 30),
         ("teste_tamanho", "w", [6, 7], 30), ("teste_tamanho", "qft", [6, 7], 30),
         ("teste_tamanho", "grover", [6, 7], 30), ("teste_estrutura", "hea", [2, 3, 4, 5, 6, 7], 200)]
PROMPT = ("Traduza este circuito Qiskit para um kernel CUDA-Q equivalente. Escreva um kernel chamado "
          "`kernel`, sem argumentos, que prepare o mesmo estado, sem medição.\n\n```python\n{q}```")


def verifica(par):
    ns = {}
    exec(par["qiskit"], ns)                    # código gerado por nós, não pelo modelo
    ref = Statevector(ns["qc"]).data
    res, psi = executa(par["cudaq"])
    fid = fidelidade(psi, ref) if psi is not None else None
    return dict(par, status=res["status"], fidelidade=fid, erro=res.get("erro"))


def sorteia(r, divisao, familia, tamanhos, pedido):
    vistos, pares, tent = set(), [], 0
    while len(pares) < pedido and tent < pedido * 30:
        tent += 1
        n = r.choice(tamanhos)
        q, c, params = FAMILIAS[familia](r, n)
        h = hashlib.sha256(q.encode()).hexdigest()[:16]
        if h in vistos:
            continue
        vistos.add(h)
        pares.append(dict(id=h, divisao=divisao, familia=familia, n=n, params=params, qiskit=q, cudaq=c))
    return pares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracao", type=float, default=1.0)
    a = ap.parse_args()
    r = random.Random(SEMENTE)
    todos = []
    for div, fam, tam, qtd in PLANO:
        pares = sorteia(r, div, fam, tam, max(1, int(qtd * a.fracao)))
        print(f"{div:16s}{fam:10s} pedidos {qtd:5d} únicos {len(pares)}", flush=True)
        todos += pares
    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(verifica, todos))
    bons = [p for p in res if p["status"] == "ok" and p["fidelidade"] >= LIMIAR]
    ruins = [p for p in res if p not in bons]
    print(f"verificados {len(res)} em {time.time() - t0:.0f} s; mantidos {len(bons)}; rejeitados {len(ruins)}")
    print("rejeições:", Counter((p["familia"], p["status"]) for p in ruins))
    for p in ruins[:3]:
        print(p["familia"], p["n"], p["status"], p["fidelidade"], (p["erro"] or "")[:200])
    print("fidelidade mínima:", min((p["fidelidade"] for p in bons), default=None))
    print("por divisão:", Counter((p["divisao"], p["familia"]) for p in bons))
    if a.fracao == 1.0:
        (RAIZ / "data").mkdir(exist_ok=True)
        for div in sorted({p["divisao"] for p in bons}):
            with open(RAIZ / "data" / f"{div}.jsonl", "w", encoding="utf8") as f:
                for p in bons:
                    if p["divisao"] == div:
                        msg = PROMPT.format(q=p["qiskit"])
                        f.write(json.dumps({k: p[k] for k in ("id", "familia", "n", "params", "fidelidade", "qiskit", "cudaq")}
                                           | {"prompt": msg, "completion": "```python\n" + p["cudaq"] + "```"},
                                           ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
