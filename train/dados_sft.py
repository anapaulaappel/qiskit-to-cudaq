"""Converte data/treino.jsonl para o formato de conversa do TRL e separa a validação.

Saída: data/sft_treino.jsonl (90%) e data/sft_validacao.jsonl (10%, estratificado por família,
semente fixa). A validação vem da mesma distribuição do treino (n = 2..5, famílias 1..5) e serve
para acompanhar a perda e escolher o checkpoint. Os conjuntos de teste nunca entram nessa escolha.

Por que conversa e não texto: com "prompt"/"completion" em texto, o TRL 0.22.2 tokeniza o texto cru,
sem chat template (sft_trainer.py, função tokenize, ramo não conversacional). A geração usa o chat
template com enable_thinking=False. Treino e avaliação ficariam em formatos diferentes.

No formato de conversa, o TRL aplica o chat template e repassa `chat_template_kwargs` de cada
exemplo. O enunciado é o mesmo campo "prompt" usado na avaliação (gerar_modal.py), sem alteração.
A conferência dos tokens está em train/confere_tokens.py.
"""
import json, random
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
KWARGS = {"enable_thinking": False}
SEMENTE, FRACAO_VALIDACAO = 20261105, 0.10
FRACAO_REPLAY = 0.20          # exemplos CUDA-Q -> Qiskit, como fração do total de treino
PROMPT_INVERSO = ("Traduza este kernel CUDA-Q para um circuito Qiskit equivalente. Escreva um circuito "
                  "chamado `qc`, sem medição.\n\n```python\n{c}```")


def para_conversa(t):
    return {"id": t["id"], "familia": t["familia"], "n": t["n"],
            "prompt": [{"role": "user", "content": t["prompt"]}],
            "completion": [{"role": "assistant", "content": t["completion"]}],
            "chat_template_kwargs": KWARGS}


def inverso(t):
    """Replay: a mesma dupla na direção CUDA-Q -> Qiskit. Mantém a sintaxe do Qiskit em uso no treino,
    para reduzir a interferência observada nos QuantumKatas (o modelo passou a escrever portas no estilo
    CUDA-Q dentro de código Qiskit). Nada dos Katas entra no treino."""
    return {"id": "inv_" + t["id"], "familia": t["familia"], "n": t["n"],
            "prompt": [{"role": "user", "content": PROMPT_INVERSO.format(c=t["cudaq"])}],
            "completion": [{"role": "assistant", "content": "```python\n" + t["qiskit"] + "```"}],
            "chat_template_kwargs": KWARGS}


def grava(nome, linhas):
    with open(RAIZ / "data" / nome, "w", encoding="utf8") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas)


def main():
    with open(RAIZ / "data" / "treino.jsonl", encoding="utf8") as f:
        linhas = [para_conversa(json.loads(l)) for l in f]
    r = random.Random(SEMENTE)
    por_familia = defaultdict(list)
    for l in linhas:
        por_familia[l["familia"]].append(l)
    treino, validacao = [], []
    for fam in sorted(por_familia):
        grupo = por_familia[fam][:]
        r.shuffle(grupo)
        k = round(len(grupo) * FRACAO_VALIDACAO)
        validacao += grupo[:k]
        treino += grupo[k:]
    r.shuffle(treino)
    assert not {l["id"] for l in treino} & {l["id"] for l in validacao}
    grava("sft_treino.jsonl", treino)
    grava("sft_validacao.jsonl", validacao)
    print(f"treino {len(treino)} | validação {len(validacao)}",
          dict(Counter(l["familia"] for l in validacao)))

    # Variante com replay (segundo experimento): treino direto + 20% de exemplos inversos, tirados só
    # dos pares de treino (nunca da validação). A validação continua só na direção direta.
    originais = {json.loads(l)["id"]: json.loads(l) for l in open(RAIZ / "data" / "treino.jsonl", encoding="utf8")}
    ids_treino = [l["id"] for l in treino]
    n_inv = round(len(treino) * FRACAO_REPLAY / (1 - FRACAO_REPLAY))
    escolhidos = random.Random(SEMENTE + 1).sample(ids_treino, n_inv)
    com_replay = treino + [inverso(originais[i]) for i in escolhidos]
    random.Random(SEMENTE + 2).shuffle(com_replay)
    grava("sft_treino_replay.jsonl", com_replay)
    print(f"treino com replay {len(com_replay)} ({n_inv} inversos, {100 * n_inv / len(com_replay):.0f}%)")

    # Experimento A (estilos): 30% dos pares de treino com o Qiskit reescrito (generate/estilo_treino.py,
    # sem os idiomas append e funcao). Os reescritos SUBSTITUEM os originais: mesmo tamanho de treino.
    caminho = RAIZ / "data" / "treino_estilos.jsonl"
    if caminho.exists():
        estilos = {json.loads(l)["id_original"]: json.loads(l) for l in open(caminho, encoding="utf8")}
        com_estilos = [para_conversa(estilos[l["id"]]) if l["id"] in estilos else l for l in treino]
        grava("sft_treino_estilos.jsonl", com_estilos)
        print(f"treino com estilos {len(com_estilos)} ({sum(l['id'] in estilos for l in treino)} reescritos)")

    # Experimento B (replay em formato de função): treino direto + exemplos de função Qiskit
    # (generate/replay_funcao.py), mesma quantidade do primeiro replay.
    caminho = RAIZ / "data" / "replay_funcao.jsonl"
    if caminho.exists():
        funcoes = [json.loads(l) for l in open(caminho, encoding="utf8")]
        com_funcao = treino + [{"id": f["id"], "familia": f["familia"], "n": f["n"],
                                "prompt": [{"role": "user", "content": f["prompt"]}],
                                "completion": [{"role": "assistant", "content": f["completion"]}],
                                "chat_template_kwargs": KWARGS} for f in funcoes]
        random.Random(SEMENTE + 3).shuffle(com_funcao)
        grava("sft_treino_funcao.jsonl", com_funcao)
        print(f"treino com replay de função {len(com_funcao)} ({len(funcoes)} funções)")


if __name__ == "__main__":
    main()
