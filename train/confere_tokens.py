"""Confere, na CPU e sem carregar o modelo, que o treino vê os mesmos tokens que a geração.

Usa o pré-processamento do próprio TRL (SFTTrainer._prepare_dataset) e o collator do treino, com
as versões do Modal (transformers 4.56.2, trl 0.22.2). Rodar com a venv de treino:
    .venv-treino/bin/python train/confere_tokens.py

Checa, em todos os exemplos de treino e validação:
  1. o começo da sequência de treino é idêntico aos ids que gerar_modal.py dá ao modelo;
  2. nada do enunciado entra na perda, e o código inteiro entra;
  3. a resposta termina em <|im_end|> e esse token entra na perda (o modelo aprende a parar);
  4. nenhum exemplo passa de max_length (truncar cortaria o fim da resposta).
Sai com erro se alguma checagem falhar.
"""
import json, sys
from collections import Counter
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

RAIZ = Path(__file__).resolve().parent.parent
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MAX_LENGTH = 1024


def ids_da_geracao(tok, prompt):
    """Igual a eval/gerar_modal.py."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token          # padrão do projeto
    tok.padding_side = "right"
    fim = tok.convert_tokens_to_ids("<|im_end|>")

    linhas = []
    for nome in ("sft_treino.jsonl", "sft_validacao.jsonl", "sft_treino_replay.jsonl",
                 "sft_treino_estilos.jsonl", "sft_treino_funcao.jsonl"):
        with open(RAIZ / "data" / nome, encoding="utf8") as f:
            linhas += [json.loads(l) for l in f]
    ds = Dataset.from_list([{k: l[k] for k in ("prompt", "completion", "chat_template_kwargs")} for l in linhas])
    args = SFTConfig(output_dir="/tmp/confere_tokens", max_length=MAX_LENGTH, completion_only_loss=True,
                     packing=False, dataset_num_proc=1, report_to=[])
    proc = SFTTrainer._prepare_dataset(None, ds, tok, args, False, None, "train")

    falhas, tamanhos = Counter(), []
    for i, (l, ex) in enumerate(zip(linhas, proc)):
        ids, mask = ex["input_ids"], ex["completion_mask"]
        ger = ids_da_geracao(tok, l["prompt"][0]["content"])
        n_enunciado = mask.index(1)
        tamanhos.append(len(ids))
        if ids[:len(ger)] != ger:
            falhas["prefixo_difere_da_geracao"] += 1
        if any(mask[:n_enunciado]) or not all(mask[n_enunciado:]):
            falhas["mascara_nao_contigua"] += 1
        # O que o modelo gera depois do prompt de geração: código e depois <|im_end|>.
        gerado = tok.decode(ids[len(ger):])
        codigo = l["completion"][0]["content"].strip()
        if not gerado.startswith(codigo + "<|im_end|>"):
            falhas["resposta_nao_e_codigo_mais_fim"] += 1
        if fim not in ids[len(ger):] or not mask[ids.index(fim, len(ger))]:
            falhas["fim_fora_da_perda"] += 1
        if len(ids) >= MAX_LENGTH:
            falhas["truncado"] += 1

    # O collator do treino: padding vira -100 nos labels, sem apagar o <|im_end|> verdadeiro (pad = eos).
    col = DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id, completion_only_loss=True)
    lote = col([proc[0], proc[1]])
    for j in range(2):
        lab = lote["labels"][j].tolist()
        real = [x for x in lab if x != -100]
        if real[-2:] != proc[j]["input_ids"][-2:]:
            falhas["collator_apagou_fim"] += 1

    ex0 = proc[0]
    ger0 = ids_da_geracao(tok, linhas[0]["prompt"][0]["content"])
    print("Exemplo 0, tokens na perda (completion_mask = 1):")
    print(repr(tok.decode([t for t, m in zip(ex0["input_ids"], ex0["completion_mask"]) if m])))
    print("\nFim do prompt de geração:", repr(tok.decode(ger0[-6:])))
    print(f"\n{len(linhas)} exemplos | tokens: mín {min(tamanhos)}, máx {max(tamanhos)}, "
          f"média {sum(tamanhos) / len(tamanhos):.0f} | max_length {MAX_LENGTH}")
    print("Falhas:", dict(falhas) or "nenhuma")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
