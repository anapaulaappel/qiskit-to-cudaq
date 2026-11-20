"""Sonda de regressão nos Qiskit QuantumKatas (CC-BY-NC-SA-4.0; os dados não entram no repositório).

Cada candidato roda em subprocesso, junto só com o teste da tarefa: a solução canônica nunca é
visível para ele. Antes de medir modelo, `--auditoria` confere o conjunto no ambiente atual.
ATENÇÃO: --auditoria e --saidas executam código de terceiros. Rode só no ambiente combinado.
"""
import argparse, concurrent.futures as cf, json, re, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
from translate_eval import extrai_codigo, wilson

KATAS = RAIZ / "data" / "katas" / "qiskit_quantumkatas.jsonl"
APROVADAS = RAIZ / "results" / "katas_auditoria.json"
CHAMADA = '''
import sys as _s
_testes = [v for k, v in list(globals().items()) if k.startswith("test") and callable(v)]
if not _testes:
    print("@@sem_teste"); _s.exit(0)
for _t in _testes:
    try:
        _t()
    except AssertionError as _e:
        print("@@falhou"); print(str(_e)[:200]); _s.exit(0)
    except BaseException as _e:
        print("@@erro"); print(type(_e).__name__, str(_e)[:200]); _s.exit(0)
print("@@passou")
'''


def carrega():
    with open(KATAS, encoding="utf8") as f:
        return [json.loads(l) for l in f]


def roda(codigo, teste, timeout=120):
    """passou, falhou, erro, sintaxe, timeout, sem_teste ou falha_processo."""
    try:
        compile(codigo, "candidato", "exec")
    except (SyntaxError, ValueError):
        return "sintaxe"
    with tempfile.TemporaryDirectory() as tmp:
        arq = Path(tmp) / "kata.py"
        arq.write_text(codigo + "\n\n" + teste + "\n" + CHAMADA, encoding="utf8")
        try:
            p = subprocess.run([sys.executable, "-I", "-W", "ignore", str(arq)], cwd=tmp,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout"
    m = re.findall(r"^@@(\w+)$", p.stdout, re.M)
    if m:
        return m[-1]
    return "erro" if "Error" in p.stderr else "falha_processo"


def auditoria(repeticoes=3):
    tarefas = carrega()
    trabalhos = [(t, "canonica", t["canonical_solution"]) for t in tarefas for _ in range(repeticoes)]
    trabalhos += [(t, "esqueleto", t["prompt"]) for t in tarefas]
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda x: (x[0]["task_id"], x[1], roda(x[2], x[0]["test"])), trabalhos))
    can, esq = {}, {}
    for tid, tipo, st in res:
        if tipo == "canonica":
            can.setdefault(tid, []).append(st)
        else:
            esq[tid] = st
    estaveis = [t for t, v in can.items() if v == ["passou"] * repeticoes]
    instaveis = {t: v for t, v in can.items() if "passou" in v and v != ["passou"] * repeticoes}
    quebradas = {t: v for t, v in can.items() if "passou" not in v}
    vazias = sorted(t for t, s in esq.items() if s == "passou")   # teste que aprova código vazio
    validas = sorted(set(estaveis) - set(vazias))
    r = dict(total=len(tarefas), repeticoes=repeticoes, validas=validas, instaveis=instaveis,
             quebradas=quebradas, aprovam_esqueleto=vazias)
    APROVADAS.write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(f"total {len(tarefas)} | canônica estável {len(estaveis)} | instáveis {len(instaveis)} | "
          f"quebradas {len(quebradas)} | teste aprova esqueleto {len(vazias)} | válidas para medir {len(validas)}")
    print("canônica:", Counter(s for v in can.values() for s in v), "| esqueleto:", Counter(esq.values()))


def avalia(saidas):
    """Mede só as tarefas aprovadas pela auditoria, sempre as mesmas no antes e no depois."""
    validas = set(json.loads(APROVADAS.read_text())["validas"])
    tarefas = {t["task_id"]: t for t in carrega() if t["task_id"] in validas}
    faltando = set(tarefas) - {s["id"] for s in saidas}
    if faltando:
        raise ValueError(f"{len(faltando)} tarefas sem saída")
    pares = [(tarefas[s["id"]], extrai_codigo(s["amostras"][0])) for s in saidas if s["id"] in tarefas]
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda x: (x[0]["task_id"], roda(x[1], x[0]["test"])), pares))
    medidas = [(t, s) for t, s in res if s != "falha_processo"]
    k, n = sum(s == "passou" for _, s in medidas), len(medidas)
    return dict(n=n, falhas_ambiente=len(res) - n, aprovadas=k, taxa=k / n if n else None, ic95=wilson(k, n),
                status=dict(Counter(s for _, s in res)), por_tarefa=dict(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auditoria", action="store_true")
    ap.add_argument("--saidas")
    ap.add_argument("--resultado")
    a = ap.parse_args()
    if a.auditoria:
        auditoria()
    else:
        with open(a.saidas, encoding="utf8") as f:
            r = avalia([json.loads(l) for l in f])
        Path(a.resultado).write_text(json.dumps(r, indent=1))
        print({k: r[k] for k in ("n", "falhas_ambiente", "taxa", "ic95", "status")})
