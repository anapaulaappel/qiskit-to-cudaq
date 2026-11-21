"""Geração pelo vLLM (k8s/vllm-nemotron.yaml), igual para linha de base e pós-treino. Roda no
workbench, que alcança o Service pelo DNS interno do cluster. Não avalia: só salva o texto.

    python eval/gerar_vllm.py --nome baseline
    python eval/gerar_vllm.py --nome finetuned --modelo-servido qiskit2cudaq    # adaptador LoRA
    python eval/gerar_vllm.py --nome fewshot --fewshot 4 --tipo traducao        # base + 4 exemplos no prompt

O prompt é tokenizado aqui, com o mesmo tokenizer e o mesmo template conferidos por
train/confere_tokens.py (enable_thinking=False), e enviado ao vLLM como lista de ids em
/v1/completions. Assim o modelo recebe exatamente os tokens que o treino viu, sem depender do
template do servidor nem da versão do transformers da imagem.

Regras iguais nas duas rodadas: temperatura 0 na primeira amostra, max_tokens 1024, parada nos ids
2 (</s>) e 11 (<|im_end|>), semente por tarefa. As amostras extras (--k > 1) usam temperatura 0,7 e
top_p 0,95, independentes. Retoma de onde parou. Grava a versão do vLLM e os parâmetros no .meta.json.
Concorrência 1, de propósito: o vLLM agrupa requisições em lote, e em BF16 a saída greedy depende
da composição do lote. Medido em 05/11/2026 com 40 tarefas: concorrência 16 repetida deu saídas
idênticas em só 23 de 40; concorrência 1 repetida, em 40 de 40 (~6,5 s por tarefa na L4). Uma
medição antes/depois com concorrência alta misturaria ruído do servidor com efeito do treino.
"""
import argparse, concurrent.futures as cf, json, os, sys, threading, time, urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
URL = os.environ.get("VLLM_URL", "http://vllm-nemotron.quantum-finetune.svc:8000")
TAREFAS = RAIZ / "data" / "tarefas_modelo.jsonl"   # outro arquivo com --tarefas
SEMENTE, MAX_TOKENS, PARADA = 20261104, 1024, [2, 11]


def exemplos_por_id(ids):
    """Exemplos escolhidos explicitamente (ex.: conjunto 2 com o exemplo aleatório trocado por um que
    cobre todas as portas). Só pares do treino."""
    pares = {json.loads(l)["id"]: json.loads(l) for l in open(RAIZ / "data" / "sft_treino.jsonl", encoding="utf8")}
    msgs = []
    for i in ids:
        msgs += [pares[i]["prompt"][0], pares[i]["completion"][0]]
    return msgs, list(ids)


def exemplos_fewshot(n, conjunto=0):
    """n pares fixos do conjunto de treino (nunca de validação ou teste), um por família, na ordem
    abaixo: cobrem portas simples e controladas, laço, r1.ctrl com laços aninhados e z.ctrl com vários
    controles. Escolha determinística: o primeiro par de cada família com 3 qubits (2 na QFT, que
    tem poucos pares), no arquivo de treino já embaralhado com semente fixa. `conjunto` = k usa o
    (k+1)-ésimo par de cada família: mesma composição de famílias e tamanhos, exemplos diferentes
    (para medir a sensibilidade à escolha dos exemplos). conjunto=0 reproduz a linha de base."""
    ordem = [("aleatorio", 3), ("ghz", 3), ("qft", 2), ("grover", 3), ("w", 3)]
    pares = [json.loads(l) for l in open(RAIZ / "data" / "sft_treino.jsonl", encoding="utf8")]
    escolhidos = []
    for fam, nq in ordem[:n]:
        cand = [x for x in pares if x["familia"] == fam and x["n"] == nq and not x["id"].startswith("inv_")]
        p = cand[conjunto % len(cand)]
        escolhidos.append(p)
    msgs = []
    for p in escolhidos:
        msgs += [p["prompt"][0], p["completion"][0]]
    return msgs, [p["id"] for p in escolhidos]


def post(caminho, corpo, timeout=600):
    req = urllib.request.Request(URL + caminho, data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(caminho):
    with urllib.request.urlopen(URL + caminho, timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--modelo-servido", default="nemotron", help="nome no vLLM: 'nemotron' (base) ou o do adaptador")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--concorrencia", type=int, default=1, help="1 = reprodutível; >1 só para testes")
    ap.add_argument("--limite", type=int)
    ap.add_argument("--fewshot", type=int, default=0, help="número de exemplos resolvidos no prompt (0 = zero-shot)")
    ap.add_argument("--tipo", choices=["traducao", "kata"], help="gera só um tipo de tarefa")
    ap.add_argument("--conjunto-fewshot", type=int, default=0, help="qual conjunto de exemplos (0 = o da linha de base)")
    ap.add_argument("--prompt-sistema", help="arquivo com um prompt de sistema (ex.: o melhor prompt do GEPA)")
    ap.add_argument("--tokenizer", default=TOKENIZER, help="outro modelo base: o tokenizer dele (o vLLM precisa servir o mesmo)")
    ap.add_argument("--tarefas", default=str(TAREFAS), help="arquivo de tarefas (padrão: as 640 traduções + 349 Katas)")
    ap.add_argument("--ids", help="arquivo com um id por linha: gera só esse subconjunto")
    ap.add_argument("--exemplos-ids", help="ids de treino separados por vírgula para o prompt few-shot (substitui --fewshot)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    if a.tokenizer == TOKENIZER:
        assert tok.convert_tokens_to_ids("<|im_end|>") == 11, "tokenizer inesperado"
        parada = PARADA
    else:      # outro modelo: eos do tokenizer e os marcadores de fim de turno que existirem
        fim = [tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>")]
        parada = sorted({i for i in [tok.eos_token_id, *fim] if isinstance(i, int) and i != tok.unk_token_id})

    saida = RAIZ / "data" / "saidas" / f"{a.nome}.jsonl"
    saida.parent.mkdir(parents=True, exist_ok=True)
    feitas = set()
    if saida.exists():
        feitas = {json.loads(l)["id"] for l in open(saida, encoding="utf8")}
    tarefas = [json.loads(l) for l in open(a.tarefas, encoding="utf8")]
    if a.ids:
        manter = set(open(a.ids).read().split()); tarefas = [t for t in tarefas if t["id"] in manter]
    pendentes = [(i, t) for i, t in enumerate(tarefas) if t["id"] not in feitas and (not a.tipo or t["tipo"] == a.tipo)][: a.limite]
    if a.exemplos_ids:
        prefixo, ids_exemplos = exemplos_por_id(a.exemplos_ids.split(","))
    else:
        prefixo, ids_exemplos = exemplos_fewshot(a.fewshot, a.conjunto_fewshot) if a.fewshot else ([], [])
    sistema = Path(a.prompt_sistema).read_text(encoding="utf8").strip() if a.prompt_sistema else None
    if sistema:
        prefixo = [{"role": "system", "content": sistema}] + prefixo

    modelos = [m["id"] for m in get("/v1/models")["data"]]
    assert a.modelo_servido in modelos, f"{a.modelo_servido} não está servido; disponíveis: {modelos}"
    versao = get("/version").get("version")
    meta = {"nome": a.nome, "motor": "vllm", "vllm": versao, "modelo_servido": a.modelo_servido, "url": URL,
            "k": a.k, "max_tokens": MAX_TOKENS, "parada_ids": parada, "concorrencia": a.concorrencia,
            "tokenizer": a.tokenizer, "enable_thinking": False, "fewshot": a.fewshot or len(ids_exemplos), "exemplos_fewshot": ids_exemplos, "tarefas": Path(a.tarefas).name, "ids": a.ids,
            "conjunto_fewshot": a.conjunto_fewshot, "prompt_sistema": sistema,
            "tipo": a.tipo or "todos"}
    saida.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(meta, f"| {len(feitas)} já feitas, {len(pendentes)} pendentes", flush=True)

    def uma(item):
        i, t = item
        ids = tok.apply_chat_template(prefixo + [{"role": "user", "content": t["prompt"]}], add_generation_prompt=True,
                                      enable_thinking=False)
        amostras, truncadas = [], []
        for j in range(a.k):
            p = {"temperature": 0.0} if j == 0 else {"temperature": 0.7, "top_p": 0.95}
            r = post("/v1/completions", {"model": a.modelo_servido, "prompt": ids, "max_tokens": MAX_TOKENS,
                                         "stop_token_ids": parada, "skip_special_tokens": True,
                                         "seed": SEMENTE + i * 1000 + j, **p})
            c = r["choices"][0]
            amostras.append(c["text"])
            truncadas.append(c.get("finish_reason") == "length")
        return {"id": t["id"], "tipo": t["tipo"], "amostras": amostras, "truncadas": truncadas}

    trava, t0, n = threading.Lock(), time.time(), 0
    with open(saida, "a", encoding="utf8") as out, cf.ThreadPoolExecutor(a.concorrencia) as ex:
        for res in ex.map(uma, pendentes):
            with trava:
                out.write(json.dumps(res, ensure_ascii=False) + "\n"); out.flush()
                n += 1
                if n % 50 == 0:
                    seg = (time.time() - t0) / n
                    print(f"{n}/{len(pendentes)} feitas, {seg:.2f} s por tarefa, ~{(len(pendentes) - n) * seg / 60:.0f} min restantes", flush=True)
    print(f"fim: {n} tarefas em {(time.time() - t0) / 60:.1f} min -> {saida}")


if __name__ == "__main__":
    main()
