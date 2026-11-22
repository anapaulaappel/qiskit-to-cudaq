"""Avalia o conjunto converter-hard: o kernel candidato é chamado com cada conjunto de argumentos da tarefa
e comparado com o estado do circuito que a função Qiskit devolve com os mesmos argumentos. A tarefa conta
como equivalente só se os três conjuntos passam (fidelidade >= 0,999).
  python eval/avalia_args.py data/saidas/hard_lora.jsonl --resultado results/hard_lora.json
"""
import argparse, concurrent.futures as cf, json, sys
from collections import Counter, defaultdict
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "verify")); sys.path.insert(0, str(RAIZ / "eval"))
from qiskit.quantum_info import Statevector
from equivalence import fidelidade
from runner import executa_args
from translate_eval import extrai_codigo, wilson, LIMIAR


def tarefas():
    return {json.loads(l)["id"]: json.loads(l) for l in open(RAIZ / "data" / "teste_converter_hard.jsonl", encoding="utf8")}


def ref(t, args):
    ns = {}; exec(t["qiskit"], ns)                      # código do corpus, não do modelo
    return Statevector(ns[t["funcao"]](*args)).data


def avalia(par):
    t, texto = par
    res, estados = executa_args(extrai_codigo(texto), t["args"])
    if estados is None:
        return dict(id=t["id"], familia=t["familia"], status=res["status"], valido=False, equivalente=False, por_conjunto=[])
    f = [fidelidade(e, ref(t, a)) if e.size == 2 ** a[0] else 0.0 for e, a in zip(estados, t["args"])]
    return dict(id=t["id"], familia=t["familia"], status="ok", valido=True, equivalente=min(f) >= LIMIAR,
                por_conjunto=[x >= LIMIAR for x in f], fidelidades=f)


def resume(L):
    n = len(L); v = sum(l["valido"] for l in L); e = sum(l["equivalente"] for l in L)
    c = sum(sum(l["por_conjunto"]) for l in L)
    return dict(n=n, saida_valida=v / n, pass_1=e / n, pass_1_ic95=wilson(e, n), conjuntos_ok=c / (3 * n),
                status=dict(Counter(l["status"] for l in L)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("saidas"); ap.add_argument("--resultado", required=True)
    a = ap.parse_args()
    T = tarefas(); S = {json.loads(l)["id"]: json.loads(l)["amostras"][0] for l in open(a.saidas, encoding="utf8")}
    assert set(T) <= set(S), "faltam saídas"
    with cf.ThreadPoolExecutor(6) as ex:
        L = list(ex.map(avalia, [(T[i], S[i]) for i in T]))
    g = defaultdict(list)
    for l in L: g[l["familia"]].append(l)
    out = dict(limiar=LIMIAR, geral=resume(L), por_familia={k: resume(v) for k, v in sorted(g.items())}, por_tarefa=L)
    Path(a.resultado).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print({k: v for k, v in out["geral"].items() if k != "pass_1_ic95"})
