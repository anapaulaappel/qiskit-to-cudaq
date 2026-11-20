"""Teste de estilo: os mesmos circuitos do teste, com o lado Qiskit escrito de outro jeito.

Motivo: um tradutor por regras de ~80 linhas acerta 100% de teste_tamanho e teste_estrutura,
porque todo o corpus usa um único estilo (qc.porta(args), uma por linha). Este conjunto mede se
a tradução sobrevive a idiomas comuns do Qiskit que o treino nunca mostra.

O que muda: só a escrita do Qiskit. Portas, tamanhos, famílias e a referência CUDA-Q são os
mesmos do circuito original. Cada reescrita é executada e precisa dar exatamente o mesmo estado
do original (fidelidade >= 1 - 1e-9); o avaliador calcula a referência a partir do original
(campo qiskit_canonico).

Idiomas (cada circuito recebe de 2 a 4, sorteados; ficam em params["estilos"]):
  nome        o circuito não se chama qc
  registro    QuantumRegister e índices qr[i]
  difusao     portas de 1 qubit em vários qubits numa chamada só: h([0, 2]) ou h(range(n))
  append      objetos de porta: append(HGate(), [0]), CXGate, RYGate(θ), ...
  funcao      o corpo fica numa função auxiliar que recebe o circuito
  barreira    barrier() no meio
  medida      measure_all() no fim (o enunciado continua pedindo kernel sem medição)
  numpy       math.pi vira np.pi (QFT)
Famílias com laço (QFT e Grover) recebem só os idiomas de texto (nome, registro, numpy, medida),
para não virarem centenas de linhas.

Uso: python generate/estilo.py   ->   data/teste_estilo.jsonl
"""
import json, random, re, sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "generate"))
from build import PROMPT
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

SEMENTE, POR_ARQUIVO = 20261105, 100
NOMES = ["circ", "circuit", "qcirc", "prog", "c"]
UM = {"h", "x", "y", "z", "s", "t"}
CLASSE = {"h": "HGate", "x": "XGate", "y": "YGate", "z": "ZGate", "s": "SGate", "t": "TGate",
          "rx": "RXGate", "ry": "RYGate", "rz": "RZGate", "cx": "CXGate", "cz": "CZGate",
          "swap": "SwapGate", "cry": "CRYGate", "cp": "CPhaseGate"}
LACO = {"qft", "grover"}


def circuito(fonte):
    """Executa o código e devolve o último QuantumCircuit do namespace, sem medidas finais."""
    ns = {}
    exec(fonte, ns)                              # código gerado por nós, não pelo modelo
    qc = [v for v in ns.values() if isinstance(v, QuantumCircuit)][-1]
    # remove_final_measurements reconstrói o circuito e reordena as portas; só usar se houver medida.
    if any(i.operation.name == "measure" for i in qc.data):
        qc = qc.remove_final_measurements(inplace=False)
    return qc


def portas(qc):
    out = []
    for ins in qc.data:
        nome = ins.operation.name
        if nome == "barrier":
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        out.append((nome, [float(p) for p in ins.operation.params], qs))
    return out


def fmt(p):
    return repr(round(p, 6))


def emite_plano(n, gs, est, r):
    """Reescreve uma lista de portas plana com os idiomas pedidos."""
    nome = r.choice(NOMES) if "nome" in est else "qc"
    q = (lambda i: f"qr[{i}]") if "registro" in est else str
    alvo = "circ_" if "funcao" in est else nome
    corpo = []
    i = 0
    while i < len(gs):
        g, ps, qs = gs[i]
        if "difusao" in est and g in UM:          # junta a mesma porta em qubits diferentes
            j, grupo = i, []
            while j < len(gs) and gs[j][0] == g and gs[j][2][0] not in grupo:
                grupo.append(gs[j][2][0]); j += 1
            if len(grupo) > 1:
                arg = (f"range({n})" if sorted(grupo) == list(range(n)) and "registro" not in est
                       else "[" + ", ".join(q(x) for x in grupo) + "]")
                corpo.append(f"{alvo}.{g}({arg})"); i = j; continue
        if g == "mcx":
            ctrl = "[" + ", ".join(q(x) for x in qs[:-1]) + "]"
            corpo.append(f"{alvo}.mcx({ctrl}, {q(qs[-1])})")
        elif "append" in est:
            args = ", ".join(fmt(p) for p in ps)
            corpo.append(f"{alvo}.append({CLASSE[g]}({args}), [{', '.join(q(x) for x in qs)}])")
        else:
            args = [fmt(p) for p in ps] + [q(x) for x in qs]
            corpo.append(f"{alvo}.{g}({', '.join(args)})")
        if "barreira" in est and r.random() < 0.08:
            corpo.append(f"{alvo}.barrier()")
        i += 1
    imp = ["from qiskit import QuantumCircuit" + (", QuantumRegister" if "registro" in est else "")]
    if "append" in est:
        usadas = sorted({CLASSE[g] for g, _, _ in gs if g in CLASSE})
        imp.append("from qiskit.circuit.library import " + ", ".join(usadas))
    linhas = imp + [""]
    if "registro" in est:
        linhas += [f'qr = QuantumRegister({n}, "q")']
    cria = f"QuantumCircuit(qr)" if "registro" in est else f"QuantumCircuit({n})"
    if "funcao" in est:
        args = "circ_, qr" if "registro" in est else "circ_"
        linhas += ["", f"def monta({args}):", *["    " + l for l in corpo], "", "",
                   f"{nome} = {cria}", f"monta({nome}{', qr' if 'registro' in est else ''})"]
    else:
        linhas += [f"{nome} = {cria}", *corpo]
    if "medida" in est:
        linhas.append(f"{nome}.measure_all()")
    return "\n".join(linhas) + "\n"


def emite_texto(fonte, n, est, r):
    """Idiomas de texto para famílias com laço: o código original com troca de nomes."""
    s = fonte
    if "nome" in est:
        s = re.sub(r"\bqc\b", r.choice(NOMES), s)
    nome = re.search(r"^(\w+) = QuantumCircuit", s, re.M)[1]
    if "registro" in est:
        s = s.replace("from qiskit import QuantumCircuit", "from qiskit import QuantumCircuit, QuantumRegister")
        s = s.replace(f"{nome} = QuantumCircuit({n})", f'qr = QuantumRegister({n}, "q")\n{nome} = QuantumCircuit(qr)')
    if "numpy" in est and "math.pi" in s:
        s = s.replace("import math\n", "import numpy as np\n").replace("math.pi", "np.pi")
    if "medida" in est:
        s = s.rstrip("\n") + f"\n{nome}.measure_all()\n"
    return s


def main():
    r = random.Random(SEMENTE)
    tarefas = []
    for arq in ("teste_tamanho", "teste_estrutura"):
        with open(RAIZ / "data" / f"{arq}.jsonl", encoding="utf8") as f:
            todas = [json.loads(l) for l in f]
        tarefas += r.sample(todas, POR_ARQUIVO)

    saida, rejeitados = [], 0
    for t in tarefas:
        if t["familia"] in LACO:
            est = sorted(r.sample(["nome", "registro", "numpy", "medida"], r.randint(2, 4)))
            q = emite_texto(t["qiskit"], t["n"], est, r)
        else:
            est = sorted(r.sample(["nome", "registro", "difusao", "append", "funcao", "barreira", "medida"],
                                  r.randint(2, 4)))
            q = emite_plano(t["n"], portas(circuito(t["qiskit"])), est, r)
        ref = Statevector(circuito(t["qiskit"])).data
        novo = Statevector(circuito(q)).data
        fid = abs(novo.conj() @ ref) ** 2 if novo.size == ref.size else 0.0
        if fid < 1 - 1e-9:
            rejeitados += 1
            print("REJEITADO", t["id"], est, fid)
            continue
        saida.append({"id": "estilo_" + t["id"], "id_original": t["id"], "familia": t["familia"], "n": t["n"],
                      "params": t["params"] | {"estilos": est}, "fidelidade": fid,
                      "qiskit": q, "qiskit_canonico": t["qiskit"], "cudaq": t["cudaq"],
                      "prompt": PROMPT.format(q=q), "completion": t["completion"]})
    with open(RAIZ / "data" / "teste_estilo.jsonl", "w", encoding="utf8") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in saida)
    print(f"{len(saida)} tarefas em data/teste_estilo.jsonl, {rejeitados} rejeitadas")
    print("famílias:", dict(Counter(s["familia"] for s in saida)))
    print("idiomas:", dict(Counter(e for s in saida for e in s["params"]["estilos"])))


if __name__ == "__main__":
    main()
