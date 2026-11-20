"""Compara linha de base e pós-treino nas mesmas tarefas (comparação pareada).

    python eval/compara.py                          # baseline x finetuned -> results/comparacao.json
    python eval/compara.py finetuned replay         # quaisquer dois -> results/comparacao_finetuned_replay.json

Lê results/{antes,depois}_traducao.json e results/{antes,depois}_katas.json.
Para cada grupo: taxa antes e depois com intervalo de Wilson, e teste de McNemar exato sobre os
pares discordantes (tarefas que só passaram antes ou só depois). Comparar a sobreposição de dois
intervalos de Wilson seria conservador demais, porque as tarefas são as mesmas nos dois lados.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
from translate_eval import wilson


def mcnemar_exato(b, c):
    """p bilateral do teste binomial exato com n = b + c pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def resumo(pares):
    """pares: lista de (antes: bool, depois: bool)."""
    n = len(pares)
    a = sum(x for x, _ in pares); d = sum(y for _, y in pares)
    so_antes = sum(x and not y for x, y in pares); so_depois = sum(y and not x for x, y in pares)
    return {"n": n, "antes": a / n, "antes_ic95": wilson(a, n), "depois": d / n, "depois_ic95": wilson(d, n),
            "so_antes": so_antes, "so_depois": so_depois, "mcnemar_p": mcnemar_exato(so_antes, so_depois)}


def main():
    antes, depois = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("baseline", "finetuned")
    out = {"antes": antes, "depois": depois}
    trad = {}
    for lado in (antes, depois):
        r = json.loads((RAIZ / "results" / f"{lado}_traducao.json").read_text())
        trad[lado] = {t["id"]: t for t in r["por_tarefa"]}
    grupos = defaultdict(lambda: {"equivalencia": [], "valida": []})
    for tid, b in trad[antes].items():
        f = trad[depois][tid]
        ab, af = b["amostras"][0], f["amostras"][0]
        if ab["valido"] is None or af["valido"] is None:
            continue                                   # falha de ambiente em um dos lados
        for g in ("todas", b["divisao"], f'{b["divisao"]}/{b["familia"]}'):
            grupos[g]["equivalencia"].append((bool(ab["equivalente"]), bool(af["equivalente"])))
            grupos[g]["valida"].append((bool(ab["valido"]), bool(af["valido"])))
    out["traducao"] = {g: {m: resumo(p) for m, p in v.items()} for g, v in sorted(grupos.items())}

    arquivos_k = {lado: RAIZ / "results" / f"{lado}_katas.json" for lado in (antes, depois)}
    tem_katas = all(f.exists() for f in arquivos_k.values())
    k = {lado: json.loads(f.read_text())["por_tarefa"] for lado, f in arquivos_k.items()} if tem_katas else {}
    if tem_katas:
        pares = [(k[antes][t] == "passou", k[depois][t] == "passou")
                 for t in k[antes] if k[antes][t] != "falha_processo" and k[depois].get(t, "falha_processo") != "falha_processo"]
        out["katas"] = resumo(pares)
    nome = "comparacao.json" if (antes, depois) == ("baseline", "finetuned") else f"comparacao_{antes}_{depois}.json"
    (RAIZ / "results" / nome).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    pc = lambda x: f"{100 * x:5.1f}%"
    ic = lambda t: f"[{100 * t[0]:.1f}–{100 * t[1]:.1f}]"
    print(f"{'grupo':34s} {'métrica':12s} {'antes':>7s} {'':15s} {'depois':>7s} {'':15s} {'só antes':>8s} {'só depois':>9s} {'McNemar p':>10s}")
    for g, v in out["traducao"].items():
        for m in ("valida", "equivalencia"):
            s = v[m]
            print(f"{g:34s} {m:12s} {pc(s['antes'])} {ic(s['antes_ic95']):15s} {pc(s['depois'])} {ic(s['depois_ic95']):15s} "
                  f"{s['so_antes']:8d} {s['so_depois']:9d} {s['mcnemar_p']:10.2g}")
    if not tem_katas:
        return
    s = out["katas"]
    print(f"{'QuantumKatas (regressão)':34s} {'passou':12s} {pc(s['antes'])} {ic(s['antes_ic95']):15s} {pc(s['depois'])} {ic(s['depois_ic95']):15s} "
          f"{s['so_antes']:8d} {s['so_depois']:9d} {s['mcnemar_p']:10.2g}")


if __name__ == "__main__":
    main()
