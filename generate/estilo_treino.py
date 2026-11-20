"""Experimento A: pares de TREINO com o Qiskit reescrito em outros estilos.

Usa as mesmas funções do teste de estilo (generate/estilo.py), mas só sobre pares de treino e só com
6 dos 8 idiomas. `append` e `funcao` ficam reservados para o teste: no teste de estilo, as tarefas que
contêm algum deles medem generalização para idioma nunca visto; as outras medem idioma visto no treino.

A validação não muda (pares da validação ficam de fora). Cada reescrita é executada e precisa dar o
mesmo estado do original. Saída: data/treino_estilos.jsonl, no mesmo formato de data/treino.jsonl.
    python generate/estilo_treino.py
"""
import json, random, sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "generate"))
from build import PROMPT
from estilo import LACO, circuito, emite_plano, emite_texto, portas
from qiskit.quantum_info import Statevector

SEMENTE, FRACAO = 20261106, 0.30
VISTOS_PLANO = ["nome", "registro", "difusao", "barreira", "medida"]     # sem append e funcao
VISTOS_TEXTO = ["nome", "registro", "numpy", "medida"]
RESERVADOS = ["append", "funcao"]


def main():
    r = random.Random(SEMENTE)
    validacao = {json.loads(l)["id"] for l in open(RAIZ / "data" / "sft_validacao.jsonl", encoding="utf8")}
    treino = [json.loads(l) for l in open(RAIZ / "data" / "treino.jsonl", encoding="utf8")]
    treino = [t for t in treino if t["id"] not in validacao]
    # Circuitos longos reescritos (ex.: qr[i] em todas as portas) passam de max_length; só até 32 portas.
    elegiveis = [t for t in treino if t["familia"] in LACO or len(portas(circuito(t["qiskit"]))) <= 32]
    escolhidos = r.sample(elegiveis, round(len(treino) * FRACAO))
    saida, rejeitados = [], 0
    for t in escolhidos:
        if t["familia"] in LACO:
            est = sorted(r.sample(VISTOS_TEXTO, r.randint(1, 3)))
            q = emite_texto(t["qiskit"], t["n"], est, r)
        else:
            est = sorted(r.sample(VISTOS_PLANO, r.randint(1, 3)))
            q = emite_plano(t["n"], portas(circuito(t["qiskit"])), est, r)
        ref, novo = Statevector(circuito(t["qiskit"])).data, Statevector(circuito(q)).data
        if novo.size != ref.size or abs(novo.conj() @ ref) ** 2 < 1 - 1e-9:
            rejeitados += 1
            continue
        assert not set(est) & set(RESERVADOS)
        saida.append(t | {"id": "est_" + t["id"], "id_original": t["id"], "qiskit": q,
                          "params": t["params"] | {"estilos": est}, "prompt": PROMPT.format(q=q)})
    with open(RAIZ / "data" / "treino_estilos.jsonl", "w", encoding="utf8") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in saida)
    print(f"{len(saida)} pares reescritos ({rejeitados} rejeitados) de {len(treino)} de treino")
    print("idiomas:", dict(Counter(e for s in saida for e in s["params"]["estilos"])))


if __name__ == "__main__":
    main()
