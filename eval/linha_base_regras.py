"""Linha de base sem modelo: tradutor Qiskit -> CUDA-Q por regras, linha a linha (~70 linhas).

Escrito para o estilo canônico do corpus (qc.porta(args), uma por linha). Acerta 100% de
teste_tamanho e teste_estrutura, o que mostra que esses dois conjuntos sozinhos não separam
"traduz" de "troca texto". No teste de estilo ele não tem regra para os idiomas novos e deixa a
linha como está. Regras novas resolveriam cada idioma; a pergunta para o modelo é se ele
generaliza sem elas. Uso: python eval/linha_base_regras.py   ->   results/linha_base_regras.json"""
import json, re, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
from translate_eval import avalia, carrega_tarefas

UM = {"h", "x", "y", "z", "s", "t"}
ROT = {"rx", "ry", "rz"}


def args(s):
    out, prof, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            prof += 1
        elif ch in ")]":
            prof -= 1
        if ch == "," and prof == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    return out + [cur.strip()] if cur.strip() else out


def q(i):
    return f"q[{i}]"


def linha(l):
    ind = re.match(r"\s*", l)[0]
    s = l.strip()
    s = re.sub(r"reversed\(range\((\w+)\)\)", r"range(\1 - 1, -1, -1)", s)
    m = re.match(r"qc\.(\w+)\((.*)\)$", s)
    if not m:
        return ind + s
    g, a = m[1], args(m[2])
    if g in UM:
        return f"{ind}{g}({q(a[0])})"
    if g in ROT:
        return f"{ind}{g}({a[0]}, {q(a[1])})"
    if g == "cx":
        return f"{ind}x.ctrl({q(a[0])}, {q(a[1])})"
    if g == "cz":
        return f"{ind}z.ctrl({q(a[0])}, {q(a[1])})"
    if g == "swap":
        return f"{ind}swap({q(a[0])}, {q(a[1])})"
    if g == "cp":
        return f"{ind}r1.ctrl({a[0]}, {q(a[1])}, {q(a[2])})"
    if g == "cry":
        return f"{ind}ry.ctrl({a[0]}, {q(a[1])}, {q(a[2])})"
    if g == "mcx":
        ctrl = [c.strip() for c in a[0].strip("[]").split(",")]
        return f"{ind}x.ctrl({', '.join(q(c) for c in ctrl)}, {q(a[1])})"
    return ind + s  # sem regra: deixa a linha como está


def traduz(src):
    corpo, extra, n = [], [], None
    for l in src.splitlines():
        if not l.strip() or l.startswith("from qiskit"):
            continue
        if l.startswith("import "):
            extra.append(l); continue
        m = re.match(r"qc = QuantumCircuit\((\d+)\)", l)
        if m:
            n = m[1]; continue
        corpo.append("    " + linha(l))
    return "\n".join(["import cudaq", *extra, "", "@cudaq.kernel", "def kernel():",
                      f"    q = cudaq.qvector({n})", *corpo]) + "\n"


T = carrega_tarefas()
res = avalia([{"id": i, "amostras": ["```python\n" + traduz(t["qiskit"]) + "```"]} for i, t in T.items()])
(RAIZ / "results" / "linha_base_regras.json").write_text(json.dumps(
    {k: res[k] for k in ("limiar_fidelidade", "geral", "por_grupo")}, indent=1, ensure_ascii=False))
print("geral", {k: res["geral"][k] for k in ("n", "saida_valida", "pass_1", "pass_1_ic95")})
for g, r in res["por_grupo"].items():
    print(f"{g:32s} n={r['n']:4d} pass@1={r['pass_1']:.3f}")
