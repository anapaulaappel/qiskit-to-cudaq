"""Controle negativo: estraga traduções corretas e mede quantas o verificador reprova.
Compara com a distribuição de medidas, que não enxerga fase relativa."""
import concurrent.futures as cf, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "verify"))
import numpy as np
from qiskit.quantum_info import Statevector
from equivalence import fidelidade
from runner import executa

LIMIAR = 0.999


def linhas_corpo(c):
    return [i for i, l in enumerate(c.splitlines()) if re.match(r"\s+(\w+(\.ctrl)?)\(", l)
            and "cudaq.qvector" not in l]


def apaga_porta(c, r):
    ls = c.splitlines(); ids = [i for i in linhas_corpo(c) if not ls[i - 1].rstrip().endswith(":")]
    if not ids: return None
    del ls[r.choice(ids)]
    return "\n".join(ls) + "\n"


def inverte_angulo(c, r):
    ids = [i for i, l in enumerate(c.splitlines()) if re.search(r"\b(rx|ry|rz)\(-?\d", l)]
    if not ids: return None
    ls = c.splitlines(); i = r.choice(ids)
    ls[i] = re.sub(r"\((-?)(\d)", lambda m: "(" + ("" if m[1] else "-") + m[2], ls[i], count=1)
    return "\n".join(ls) + "\n"


def troca_controle(c, r):
    ids = [i for i, l in enumerate(c.splitlines()) if re.search(r"x\.ctrl\(q\[\d\], q\[\d\]\)", l)]
    if not ids: return None
    ls = c.splitlines(); i = r.choice(ids)
    ls[i] = re.sub(r"x\.ctrl\(q\[(\d)\], q\[(\d)\]\)", r"x.ctrl(q[\2], q[\1])", ls[i])
    return "\n".join(ls) + "\n"


def fase_final(c, r):
    """Só muda fase: Z no fim mantém todas as probabilidades, de propósito."""
    n = int(re.search(r"qvector\((\d)\)", c)[1])
    return c.rstrip("\n") + f"\n    z(q[{r.randrange(n)}])\n"


MUTANTES = {"apaga_porta": apaga_porta, "inverte_angulo": inverte_angulo,
            "troca_controle": troca_controle, "fase_final": fase_final}


def avalia(t):
    nome, p, mut = t
    ns = {}; exec(p["qiskit"], ns); ref = Statevector(ns["qc"]).data
    res, psi = executa(mut)
    if psi is None:
        return nome, p["familia"], "nao_executa", None, None
    f = fidelidade(psi, ref)
    tvd = 0.5 * float(np.abs(np.abs(psi) ** 2 - np.abs(ref) ** 2).sum()) if psi.size == ref.size else 1.0
    return nome, p["familia"], "ok", f, tvd


def main():
    r = random.Random(11)
    pares = [json.loads(l) for a in ("treino", "teste_tamanho", "teste_estrutura")
             for l in open(RAIZ / "data" / f"{a}.jsonl", encoding="utf8")]
    r.shuffle(pares)
    tarefas = []
    cont = Counter()
    for p in pares:
        for nome, f in MUTANTES.items():
            if cont[nome] >= 150: continue
            m = f(p["cudaq"], r)
            if m and m != p["cudaq"]:
                tarefas.append((nome, p, m)); cont[nome] += 1
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(avalia, tarefas))
    tab = defaultdict(lambda: Counter())
    for nome, fam, st, f, tvd in res:
        c = tab[nome]; c["n"] += 1
        if st != "ok": c["nao_executa"] += 1; continue
        c["fid_pega"] += f < LIMIAR
        c["contagem_pega"] += tvd > 0.01
        c["so_fid"] += f < LIMIAR and tvd <= 0.01
        c["nao_pego"] += f >= LIMIAR
    for nome, c in tab.items():
        print(f"{nome:15s} n={c['n']:3d} não_executa={c['nao_executa']:3d} fidelidade_pega={c['fid_pega']:3d} "
              f"contagem_pega={c['contagem_pega']:3d} só_fidelidade={c['so_fid']:3d} escapou={c['nao_pego']:3d}")
    esc = [(n, fam, f) for n, fam, st, f, _ in res if st == "ok" and f >= LIMIAR]
    identicos = [e for e in esc if e[2] >= 1 - 1e-9]
    limite = sorted(e[2] for e in esc if e[2] < 1 - 1e-9)
    print(f"escapes {len(esc)}: estado idêntico {len(identicos)}; diferente mas acima do limiar {len(limite)}", limite[:12])
    so_baixos = [f for _, _, st, f, _ in res if st == "ok" and f < LIMIAR]
    print("reprovados mais perto do limiar:", [round(x, 5) for x in sorted(so_baixos)[-6:]])
    (RAIZ / "results").mkdir(exist_ok=True)
    json.dump({k: dict(v) for k, v in tab.items()}, open(RAIZ / "results" / "controle_negativo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
