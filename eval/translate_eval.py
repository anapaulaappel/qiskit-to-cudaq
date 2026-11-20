"""Avalia saídas do modelo. Lê as saídas de um JSONL e não chama modelo nenhum.

Entrada, uma linha por tarefa: {"id": ..., "amostras": ["texto do modelo", ...]}
A primeira amostra é a de do_sample=False (pass@1). As seguintes, se houver, são amostras
independentes para pass@k. O que é checado: extrai o código, executa e compara a fidelidade
com o estado do Qiskit. A referência fica neste processo; o candidato roda em subprocesso.
"""
import argparse, concurrent.futures as cf, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "verify"))
from qiskit.quantum_info import Statevector
from equivalence import fidelidade
from runner import executa

LIMIAR = 0.999
ARQUIVOS = ("teste_tamanho", "teste_estrutura", "teste_estilo")
BLOCO = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    meio = (p + z * z / (2 * n)) / den
    raio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, meio - raio), min(1.0, meio + raio))


def extrai_codigo(texto):
    """Primeiro bloco cercado que tenha kernel; senão o primeiro bloco; senão o texto todo."""
    blocos = BLOCO.findall(texto)
    return next((b for b in blocos if "cudaq" in b), blocos[0] if blocos else texto)


def carrega_tarefas():
    tarefas = {}
    for nome in ARQUIVOS:
        with open(RAIZ / "data" / f"{nome}.jsonl", encoding="utf8") as f:
            for linha in f:
                t = json.loads(linha)
                t["divisao"] = nome
                tarefas[t["id"]] = t
    return tarefas


def referencia(tarefa):
    """No teste de estilo, a referência vem do circuito original; generate/estilo.py já conferiu
    que a reescrita mostrada no enunciado dá exatamente o mesmo estado."""
    ns = {}
    exec(tarefa.get("qiskit_canonico", tarefa["qiskit"]), ns)   # código do corpus, não do modelo
    return Statevector(ns["qc"]).data


def avalia_amostra(tarefa, ref, texto):
    res, psi = executa(extrai_codigo(texto))
    if res["status"] == "falha_processo":
        return dict(status="falha_processo", valido=None, equivalente=None, fidelidade=None)
    if psi is None:
        return dict(status=res["status"], valido=False, equivalente=False, fidelidade=None)
    f = fidelidade(psi, ref)
    return dict(status="ok", valido=True, equivalente=f >= LIMIAR, fidelidade=f)


def avalia_tarefa(par):
    tarefa, amostras = par
    ref = referencia(tarefa)
    return dict(id=tarefa["id"], divisao=tarefa["divisao"], familia=tarefa["familia"], n=tarefa["n"],
                amostras=[avalia_amostra(tarefa, ref, a) for a in amostras])


def resume(linhas):
    """Taxa de saída válida e pass@1 sobre a primeira amostra; pass@k sobre todas."""
    ok = [l for l in linhas if l["amostras"][0]["valido"] is not None]
    n = len(ok)
    v = sum(l["amostras"][0]["valido"] for l in ok)
    e = sum(l["amostras"][0]["equivalente"] for l in ok)
    fids = [l["amostras"][0]["fidelidade"] for l in ok if l["amostras"][0]["fidelidade"] is not None]
    k = max((len(l["amostras"]) for l in linhas), default=0)
    r = dict(n=n, falhas_ambiente=len(linhas) - n, saida_valida=v / n if n else None, saida_valida_ic95=wilson(v, n),
             pass_1=e / n if n else None, pass_1_ic95=wilson(e, n),
             fidelidade_media_das_validas=sum(fids) / len(fids) if fids else None)
    if k > 1:
        hit = sum(any(a["equivalente"] for a in l["amostras"]) for l in ok)
        r |= {f"pass_{k}": hit / n if n else None, f"pass_{k}_ic95": wilson(hit, n)}
    r["status"] = dict(Counter(l["amostras"][0]["status"] for l in linhas))
    return r


def avalia(saidas):
    tarefas = carrega_tarefas()
    saidas = [s for s in saidas if s.get("tipo", "traducao") == "traducao"]   # Katas têm sonda própria
    faltando = set(tarefas) - {s["id"] for s in saidas}
    if faltando:
        raise ValueError(f"{len(faltando)} tarefas sem saída: uma avaliação parcial não é comparável")
    pares = [(tarefas[s["id"]], s["amostras"]) for s in saidas]
    with cf.ThreadPoolExecutor(8) as ex:
        linhas = list(ex.map(avalia_tarefa, pares))
    grupos = defaultdict(list)
    for l in linhas:
        grupos[l["divisao"]].append(l)
        grupos[f'{l["divisao"]}/{l["familia"]}'].append(l)
    return dict(limiar_fidelidade=LIMIAR, geral=resume(linhas),
                por_grupo={k: resume(v) for k, v in sorted(grupos.items())}, por_tarefa=linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("saidas")
    ap.add_argument("--resultado", required=True)
    a = ap.parse_args()
    with open(a.saidas, encoding="utf8") as f:
        saidas = [json.loads(l) for l in f]
    res = avalia(saidas)
    Path(a.resultado).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("limiar_fidelidade", "geral")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
