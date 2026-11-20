"""Geração com o modelo, igual para linha de base e pós-treino. Roda no workbench do OpenShift AI
(1 GPU L4). Não avalia: só salva o texto do modelo; a avaliação roda depois, no Mac.

    python eval/gerar.py --preparar                       # monta data/tarefas_modelo.jsonl (no Mac ou no workbench)
    python eval/gerar.py --nome baseline                   # modelo base
    python eval/gerar.py --nome finetuned --adaptador runs/lora/final

Regras que precisam ser idênticas nas duas rodadas (e são, porque é o mesmo código):
BF16, attn eager, enable_thinking=False, do_sample=False na primeira amostra, MAX_TOKENS 1024,
parada em <|im_end|> (id 11) e no eos do generation_config (id 2), semente por tarefa.
Uma tarefa por vez, sem lote: padding em lote mudaria a conta e poderia mudar a saída.
Retoma de onde parou se o arquivo de saída já existir. Com --k > 1, as amostras extras são
independentes (temperatura 0,7, top_p 0,95), para pass@k.
eval/gerar_modal.py é a alternativa na Modal, com a mesma lógica; não é a usada agora.
"""
import argparse, json, os, sys, time
from pathlib import Path

# A imagem do workbench aponta o Triton para o ptxas do CUDA 13 (TRITON_PTXAS_PATH etc.). O Triton
# 3.3.1, que acompanha o torch 2.7.1, não reconhece o CUDA 13 e quebra ao compilar os kernels do
# Mamba. Sem essas variáveis, ele usa o ptxas que vem com ele. Precisa vir antes de importar torch.
for _v in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "TRITON_CUOBJDUMP_PATH", "TRITON_NVDISASM_PATH"):
    os.environ.pop(_v, None)

RAIZ = Path(__file__).resolve().parent.parent
MODELO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
INSTRUCAO_KATA = "Complete the function below. Reply with the full Python code in one fenced block.\n\n"
SEMENTE, MAX_TOKENS = 20261104, 1024
TAREFAS = RAIZ / "data" / "tarefas_modelo.jsonl"


def preparar():
    linhas = []
    for nome in ("teste_tamanho", "teste_estrutura", "teste_estilo"):
        with open(RAIZ / "data" / f"{nome}.jsonl", encoding="utf8") as f:
            linhas += [{"id": t["id"], "tipo": "traducao", "prompt": t["prompt"]} for t in map(json.loads, f)]
    with open(RAIZ / "data" / "katas" / "qiskit_quantumkatas.jsonl", encoding="utf8") as f:
        linhas += [{"id": t["task_id"], "tipo": "kata", "prompt": INSTRUCAO_KATA + t["prompt"]}
                   for t in map(json.loads, f)]
    with open(TAREFAS, "w", encoding="utf8") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas)
    print(len(linhas), "tarefas em", TAREFAS)


def carrega(modelo, adaptador, usa_cache=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(modelo, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(modelo, torch_dtype=torch.bfloat16, device_map="auto",
                                                 trust_remote_code=True, attn_implementation="eager")
    if usa_cache:
        cache_hibrido(model)                     # antes do PeftModel, que guarda a função original
    if adaptador:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptador)
    model.eval()
    return tok, model


def cache_hibrido(model):
    """Liga o cache na geração do Nemotron-H.

    O prepare_inputs_for_generation do modeling_nemotron_h.py cria o cache híbrido (Mamba + atenção)
    mas o entrega como `past_key_values`, e o forward só lê `cache_params`. O generate do
    transformers 4.56 devolve o cache como `cache_params`, que a função original ignora, criando um
    cache vazio a cada passo. Sem cache, cada token recalcula a sequência inteira (~60 s por tarefa
    na L4). Esta versão cria o cache no primeiro passo, depois passa só os tokens novos e entrega o
    cache como `cache_params`. eval/confere_cache.py confere que a saída é a mesma sem cache."""
    import types
    Cache = sys.modules[type(model).__module__].HybridMambaAttentionDynamicCache

    def prepara(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
                cache_position=None, position_ids=None, use_cache=True, cache_params=None, **kwargs):
        primeiro = cache_params is None
        if primeiro:
            cache_params = Cache(self.config, input_ids.shape[0], self.dtype, device=self.device)
        else:
            input_ids = input_ids[:, -cache_position.shape[0]:]
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not primeiro:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        return {"input_ids": input_ids.contiguous(), "cache_params": cache_params, "use_cache": use_cache,
                "attention_mask": attention_mask, "position_ids": position_ids, "cache_position": cache_position,
                "logits_to_keep": self.config.num_logits_to_keep}

    model.prepare_inputs_for_generation = types.MethodType(prepara, model)
    return model


def ids_de_parada(tok, model):
    """O generation_config para em </s> (id 2), mas o chat template fecha a resposta com <|im_end|>
    (id 11). Sem os dois, cada amostra pode ir até MAX_TOKENS."""
    gc = model.generation_config.eos_token_id
    gc = gc if isinstance(gc, list) else [gc]
    return sorted({tok.convert_tokens_to_ids("<|im_end|>"), *[e for e in gc if e is not None]})


def gerar(nome, modelo, adaptador="", k=1, limite=None):
    import torch
    saida = RAIZ / "data" / "saidas" / f"{nome}.jsonl"
    saida.parent.mkdir(parents=True, exist_ok=True)
    feitas = set()
    if saida.exists():
        with open(saida, encoding="utf8") as f:
            feitas = {json.loads(l)["id"] for l in f}
    with open(TAREFAS, encoding="utf8") as f:
        tarefas = [json.loads(l) for l in f]
    tok, model = carrega(modelo, adaptador)
    fim = ids_de_parada(tok, model)
    meta = {"nome": nome, "modelo": modelo, "adaptador": adaptador, "k": k, "max_tokens": MAX_TOKENS,
            "parada": fim, "cache_hibrido": True, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__}
    (saida.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1))
    print(meta, f"| {len(feitas)} já feitas de {len(tarefas)}", flush=True)

    t0, feitas_agora = time.time(), 0
    with open(saida, "a", encoding="utf8") as out:
        for i, t in enumerate(tarefas):
            if t["id"] in feitas:
                continue
            if limite is not None and feitas_agora >= limite:
                break
            ids = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], add_generation_prompt=True,
                                          return_tensors="pt", enable_thinking=False).to(model.device)
            amostras, truncadas = [], []
            for j in range(k):
                torch.manual_seed(SEMENTE + i * 1000 + j)
                extra = {"do_sample": False} if j == 0 else {"do_sample": True, "temperature": 0.7, "top_p": 0.95}
                with torch.no_grad():
                    g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_TOKENS,
                                       pad_token_id=tok.eos_token_id, eos_token_id=fim, **extra)[0]
                nova = g[ids.shape[-1]:]
                amostras.append(tok.decode(nova, skip_special_tokens=True))
                truncadas.append(int(nova.shape[-1]) >= MAX_TOKENS)
            out.write(json.dumps({"id": t["id"], "tipo": t["tipo"], "amostras": amostras, "truncadas": truncadas},
                                 ensure_ascii=False) + "\n")
            out.flush()
            feitas_agora += 1
            if feitas_agora % 25 == 0:
                seg = (time.time() - t0) / feitas_agora
                falta = sum(1 for x in tarefas if x["id"] not in feitas) - feitas_agora
                print(f"{feitas_agora} feitas, {seg:.1f} s por tarefa, faltam {falta} (~{falta * seg / 3600:.1f} h)",
                      flush=True)
    print(f"fim: {feitas_agora} tarefas em {(time.time() - t0) / 60:.0f} min -> {saida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preparar", action="store_true")
    ap.add_argument("--nome")
    ap.add_argument("--modelo", default=os.environ.get("MODELO", MODELO))
    ap.add_argument("--adaptador", default="")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--limite", type=int, help="gera no máximo N tarefas (teste rápido)")
    a = ap.parse_args()
    if a.preparar:
        preparar()
    elif a.nome:
        gerar(a.nome, a.modelo, a.adaptador, a.k, a.limite)
    else:
        sys.exit("use --preparar ou --nome")
