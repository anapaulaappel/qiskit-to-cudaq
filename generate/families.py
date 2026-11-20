"""As 6 famílias de pares Qiskit/CUDA-Q. Cada função devolve (qiskit, cudaq, parametros)."""
import math

TOPO_Q = "from qiskit import QuantumCircuit\n"
TOPO_C = "import cudaq\n"
UMA = ["h", "x", "y", "z", "s", "t"]
ROT = ["rx", "ry", "rz"]
DUAS = ["cx", "cz", "swap"]


def _ang(r):
    return round(r.uniform(-math.pi, math.pi), 4)


def _qiskit(n, corpo, extra=""):
    return TOPO_Q + extra + f"\nqc = QuantumCircuit({n})\n" + "".join(l + "\n" for l in corpo)


def _cudaq(n, corpo, extra=""):
    linhas = "".join("    " + l + "\n" for l in corpo)
    return TOPO_C + extra + f"\n\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector({n})\n" + linhas


def _porta(nome, a, b=None, th=None):
    """Uma porta do conjunto fixo nos dois idiomas."""
    if nome in UMA:
        return f"qc.{nome}({a})", f"{nome}(q[{a}])"
    if nome in ROT:
        return f"qc.{nome}({th}, {a})", f"{nome}({th}, q[{a}])"
    return {"cx": (f"qc.cx({a}, {b})", f"x.ctrl(q[{a}], q[{b}])"),
            "cz": (f"qc.cz({a}, {b})", f"z.ctrl(q[{a}], q[{b}])"),
            "swap": (f"qc.swap({a}, {b})", f"swap(q[{a}], q[{b}])")}[nome]


def _par(n, portas, **params):
    return _qiskit(n, [p[0] for p in portas]), _cudaq(n, [p[1] for p in portas]), params


def aleatorio(r, n):
    d = r.randint(3, 40)
    portas = []
    for _ in range(d):
        p = r.choice(UMA + ROT + DUAS)
        if p in UMA:
            portas.append(_porta(p, r.randrange(n)))
        elif p in ROT:
            portas.append(_porta(p, r.randrange(n), th=_ang(r)))
        else:
            a, b = r.sample(range(n), 2)
            portas.append(_porta(p, a, b))
    return _par(n, portas, profundidade=d)


def ghz(r, n):
    ordem = r.sample(range(n), n)
    topo = r.choice(["cadeia", "estrela"])
    sinais = [q for q in range(n) if r.random() < 0.25]
    fase = r.choice([None, "z", "s", "t"])
    portas = [_porta("h", ordem[0])]
    for i in range(1, n):
        ctrl = ordem[i - 1] if topo == "cadeia" else ordem[0]
        portas.append(_porta("cx", ctrl, ordem[i]))
    if fase:
        portas.append(_porta(fase, ordem[0]))
    portas += [_porta("x", q) for q in sinais]
    if topo == "cadeia" and ordem == list(range(n)) and not sinais and not fase and r.random() < 0.7:
        q = ["qc.h(0)", f"for i in range({n - 1}):", "    qc.cx(i, i + 1)"]
        c = ["h(q[0])", f"for i in range({n - 1}):", "    x.ctrl(q[i], q[i + 1])"]
        return _qiskit(n, q), _cudaq(n, c), {"topologia": topo, "estilo": "laco"}
    return _par(n, portas, topologia=topo, sinais=sinais, fase=fase)


def w(r, n):
    ordem = r.sample(range(n), n)
    q = [f"qc.x({ordem[0]})"]
    c = [f"x(q[{ordem[0]}])"]
    for i in range(n - 1):
        th = round(2 * math.acos(math.sqrt(1 / (n - i))), 6)
        a, b = ordem[i], ordem[i + 1]
        q += [f"qc.cry({th}, {a}, {b})", f"qc.cx({b}, {a})"]
        c += [f"ry.ctrl({th}, q[{a}], q[{b}])", f"x.ctrl(q[{b}], q[{a}])"]
    return _qiskit(n, q), _cudaq(n, c), {"ordem": ordem}


def qft(r, n):
    troca = r.random() < 0.5
    mascara = r.randrange(2 ** n)
    q = [f"qc.x({i})" for i in range(n) if mascara >> i & 1]
    c = [f"x(q[{i}])" for i in range(n) if mascara >> i & 1]
    q += [f"for j in reversed(range({n})):", "    qc.h(j)", "    for k in reversed(range(j)):",
          "        qc.cp(math.pi / 2 ** (j - k), k, j)"]
    c += [f"for j in range({n - 1}, -1, -1):", "    h(q[j])", "    for k in range(j - 1, -1, -1):",
          "        r1.ctrl(math.pi / 2 ** (j - k), q[k], q[j])"]
    if troca:
        q += [f"for i in range({n // 2}):", f"    qc.swap(i, {n - 1} - i)"]
        c += [f"for i in range({n // 2}):", f"    swap(q[i], q[{n - 1} - i])"]
    return (_qiskit(n, q, "import math\n"), _cudaq(n, c, "import math\n"),
            {"trocas": troca, "entrada": mascara})


def grover(r, n):
    marcado = r.randrange(2 ** n)
    otimo = max(1, math.floor(math.pi / 4 * math.sqrt(2 ** n)))
    it = r.randint(1, otimo)
    zeros = [i for i in range(n) if not marcado >> i & 1]
    alvo, ctrl = n - 1, list(range(n - 1))
    args_c = ", ".join(f"q[{i}]" for i in ctrl)
    q, c = [f"for i in range({n}):", "    qc.h(i)"], [f"h(q)"]
    q += [f"for _ in range({it}):"]
    c += [f"for _ in range({it}):"]
    def mcz(qq, cc):
        qq += [f"    qc.h({alvo})", f"    qc.mcx({ctrl}, {alvo})", f"    qc.h({alvo})"]
        cc += [f"    z.ctrl({args_c}, q[{alvo}])"]
    for i in zeros:
        q.append(f"    qc.x({i})"); c.append(f"    x(q[{i}])")
    mcz(q, c)
    for i in zeros:
        q.append(f"    qc.x({i})"); c.append(f"    x(q[{i}])")
    q += [f"    for i in range({n}):", "        qc.h(i)", "        qc.x(i)"]
    c += ["    h(q)", "    x(q)"]
    mcz(q, c)
    q += [f"    for i in range({n}):", "        qc.x(i)", "        qc.h(i)"]
    c += ["    x(q)", "    h(q)"]
    return _qiskit(n, q), _cudaq(n, c), {"marcado": marcado, "iteracoes": it}


def hea(r, n):
    camadas = r.randint(1, 3)
    emaranha = r.choice(["cx", "cz"])
    circular = r.random() < 0.5 and n > 2
    portas = []
    for _ in range(camadas):
        for i in range(n):
            portas.append(_porta("ry", i, th=_ang(r)))
            portas.append(_porta("rz", i, th=_ang(r)))
        for i in range(n - 1):
            portas.append(_porta(emaranha, i, i + 1))
        if circular:
            portas.append(_porta(emaranha, n - 1, 0))
    return _par(n, portas, camadas=camadas, emaranhamento=emaranha, circular=circular)


FAMILIAS = {"aleatorio": aleatorio, "ghz": ghz, "w": w, "qft": qft, "grover": grover, "hea": hea}
