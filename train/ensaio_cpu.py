"""Ensaio do treino na CPU, antes de gastar GPU. Não mede qualidade nenhuma.

Usa o tokenizer e o chat template do Nemotron com um modelo minúsculo de pesos aleatórios (o
Nemotron precisa de mamba_ssm, que só roda em CUDA). Confere que a combinação TRL 0.22.2 +
transformers 4.56.2 + PEFT aceita a configuração de train/config.py de ponta a ponta: dados de
conversa, avaliação na validação, checkpoints, escolha do melhor e gravação do adaptador.
    .venv-treino/bin/python train/ensaio_cpu.py
"""
import json, shutil, sys, tempfile
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

RAIZ = Path(__file__).resolve().parent.parent
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
COLUNAS = ("prompt", "completion", "chat_template_kwargs")


def carrega(nome, n):
    with open(RAIZ / "data" / nome, encoding="utf8") as f:
        linhas = [json.loads(l) for l in f][:n]
    return Dataset.from_list([{k: l[k] for k in COLUNAS} for l in linhas])


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    modelo = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                          num_hidden_layers=2, num_attention_heads=4,
                                          max_position_embeddings=1024))
    saida = Path(tempfile.mkdtemp(prefix="ensaio_lora_"))
    try:
        args = config.sft(str(saida), bf16=False, optim="adamw_torch", max_steps=6, eval_steps=2,
                          save_steps=2, logging_steps=1, warmup_ratio=0.0, gradient_checkpointing=False,
                          use_cpu=True, dataloader_pin_memory=False)
        tr = SFTTrainer(model=modelo, args=args, train_dataset=carrega("sft_treino.jsonl", 48),
                        eval_dataset=carrega("sft_validacao.jsonl", 8), processing_class=tok,
                        peft_config=config.lora())
        tr.model.print_trainable_parameters()
        tr.train()
        perdas = [h["eval_loss"] for h in tr.state.log_history if "eval_loss" in h]
        melhor = tr.state.best_model_checkpoint
        tr.save_model(str(saida / "final"))
        gravados = sorted(p.name for p in (saida / "final").iterdir())
        print("perdas de validação:", [round(p, 3) for p in perdas])
        print("melhor checkpoint:", melhor and Path(melhor).name, "| arquivos do adaptador:", gravados)
        ok = len(perdas) == 3 and melhor and "adapter_model.safetensors" in gravados
        print("ENSAIO OK" if ok else "ENSAIO FALHOU")
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(saida, ignore_errors=True)


if __name__ == "__main__":
    main()
