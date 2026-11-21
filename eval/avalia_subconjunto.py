"""Avalia saídas em um conjunto de tarefas fora dos três testes fixos (ex.: data/teste_qedc.jsonl) ou em
um subconjunto deles (--ids). Mesmo verificador e mesmo limiar do translate_eval.

  python eval/avalia_subconjunto.py data/saidas/qedc_lora.jsonl --tarefas data/teste_qedc.jsonl --resultado results/qedc_lora.json
  python eval/avalia_subconjunto.py data/saidas/k5_lora.jsonl --ids data/ids_idiomas_reservados.txt --resultado results/k5_lora.json
"""
import argparse, concurrent.futures as cf, json, sys
from collections import defaultdict
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
from translate_eval import avalia_tarefa, carrega_tarefas, resume, LIMIAR

ap = argparse.ArgumentParser()
ap.add_argument("saidas"); ap.add_argument("--resultado", required=True)
ap.add_argument("--tarefas", help="jsonl com id, familia, n, qiskit[_canonico]")
ap.add_argument("--ids", help="arquivo com um id por linha (subconjunto dos testes fixos)")
a = ap.parse_args()
if a.tarefas:
    T = {}
    for l in open(a.tarefas, encoding="utf8"):
        t = json.loads(l); t["divisao"] = Path(a.tarefas).stem; T[t["id"]] = t
else:
    T = carrega_tarefas()
if a.ids:
    ids = set(open(a.ids).read().split()); T = {k: v for k, v in T.items() if k in ids}
S = {json.loads(l)["id"]: json.loads(l) for l in open(a.saidas, encoding="utf8")}
falta = set(T) - set(S)
assert not falta, f"{len(falta)} tarefas sem saída"
with cf.ThreadPoolExecutor(8) as ex:
    linhas = list(ex.map(avalia_tarefa, [(T[i], S[i]["amostras"]) for i in T]))
g = defaultdict(list)
for l in linhas: g[l["familia"]].append(l)
out = dict(limiar_fidelidade=LIMIAR, geral=resume(linhas), por_familia={k: resume(v) for k, v in sorted(g.items())}, por_tarefa=linhas)
Path(a.resultado).write_text(json.dumps(out, indent=1, ensure_ascii=False))
r = out["geral"]; print({k: r.get(k) for k in r if k.startswith(("n", "saida", "pass"))})
