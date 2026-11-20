"""Geração na Modal, igual para linha de base e pós-treino. Não avalia: só salva o texto do modelo.

Fluxo: `python eval/gerar_modal.py --preparar` monta data/tarefas_modelo.jsonl, depois
    modal volume put nemotron-vol data/tarefas_modelo.jsonl /tarefas_modelo.jsonl
    modal run --detach eval/gerar_modal.py::gerar --nome baseline
    modal volume get nemotron-vol saidas/baseline.jsonl data/saidas/
A avaliação roda depois, fora da GPU. Com --k > 1, as amostras extras são independentes.
"""
import json, os, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
INSTRUCAO_KATA = "Complete the function below. Reply with the full Python code in one fenced block.\n\n"
SEMENTE, MAX_TOKENS = 20261104, 1024


def preparar():
    linhas = []
    for nome in ("teste_tamanho", "teste_estrutura", "teste_estilo"):
        with open(RAIZ / "data" / f"{nome}.jsonl", encoding="utf8") as f:
            linhas += [{"id": t["id"], "tipo": "traducao", "prompt": t["prompt"]} for t in map(json.loads, f)]
    with open(RAIZ / "data" / "katas" / "qiskit_quantumkatas.jsonl", encoding="utf8") as f:
        linhas += [{"id": t["task_id"], "tipo": "kata", "prompt": INSTRUCAO_KATA + t["prompt"]}
                   for t in map(json.loads, f)]
    with open(RAIZ / "data" / "tarefas_modelo.jsonl", "w", encoding="utf8") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas)
    print(len(linhas), "tarefas")


if __name__ == "__main__" and "--preparar" in sys.argv:
    preparar()
    raise SystemExit

import modal
from modal_setup import MODEL_DIR, VOL, app, vol


@app.function(gpu="L4", volumes={VOL: vol}, timeout=6 * 60 * 60)
def gerar(nome: str, adaptador: str = "", k: int = 1):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saida = f"{VOL}/saidas/{nome}.jsonl"
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    feitas = set()
    if os.path.exists(saida):                       # retoma de onde parou
        with open(saida, encoding="utf8") as f:
            feitas = {json.loads(l)["id"] for l in f}

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
                                                 trust_remote_code=True, attn_implementation="eager")
    if adaptador:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, f"{VOL}/{adaptador}")
    model.eval()

    # Parar no fim de turno. O generation_config do modelo para em </s> (id 2), mas o chat template
    # fecha a resposta com <|im_end|> (id 11, o eos do tokenizer). Sem isto, cada amostra pode ir até
    # MAX_TOKENS. Treino e linha de base usam este mesmo script, então a regra vale para os dois.
    gc_eos = model.generation_config.eos_token_id
    gc_eos = gc_eos if isinstance(gc_eos, list) else [gc_eos]
    fim = sorted({tok.convert_tokens_to_ids("<|im_end|>"), *[e for e in gc_eos if e is not None]})

    with open(f"{VOL}/tarefas_modelo.jsonl", encoding="utf8") as f:
        tarefas = [json.loads(l) for l in f]
    with open(saida, "a", encoding="utf8") as out:
        for i, t in enumerate(tarefas):
            if t["id"] in feitas:
                continue
            ids = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], add_generation_prompt=True,
                                          return_tensors="pt", enable_thinking=False).to(model.device)
            amostras, truncadas = [], []
            for j in range(k):
                torch.manual_seed(SEMENTE + i * 1000 + j)
                extra = {"do_sample": False} if j == 0 else {"do_sample": True, "temperature": 0.7, "top_p": 0.95}
                g = model.generate(ids, max_new_tokens=MAX_TOKENS, pad_token_id=tok.eos_token_id,
                                   eos_token_id=fim, **extra)[0]
                nova = g[ids.shape[-1]:]
                amostras.append(tok.decode(nova, skip_special_tokens=True))
                truncadas.append(int(nova.shape[-1]) >= MAX_TOKENS)
            out.write(json.dumps({"id": t["id"], "tipo": t["tipo"], "amostras": amostras, "truncadas": truncadas},
                                 ensure_ascii=False) + "\n")
            if i % 25 == 0:
                out.flush()
                vol.commit()
    vol.commit()
