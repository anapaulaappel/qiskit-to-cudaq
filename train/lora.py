"""Treino do LoRA no workbench do OpenShift AI (1 GPU L4). Só rodar depois da linha de base.

    python train/lora.py --max-steps 4 --saida runs/ensaio    # ensaio curto na GPU
    python train/lora.py                                       # treino completo -> runs/lora

Configuração em train/config.py (a mesma do ensaio na CPU). Dados em formato de conversa
(train/dados_sft.py), conferidos por train/confere_tokens.py. O adaptador final é o de menor
perda de validação; fica em <saida>/final. A curva de perda, o tempo e a memória de pico ficam em
<saida>/historico.json, para o documento.
"""
import argparse, json, os, sys, time
from pathlib import Path

# A imagem do workbench aponta o Triton para o ptxas do CUDA 13 (TRITON_PTXAS_PATH etc.). O Triton
# 3.3.1, que acompanha o torch 2.7.1, não reconhece o CUDA 13 e quebra ao compilar os kernels do
# Mamba. Sem essas variáveis, ele usa o ptxas que vem com ele. Precisa vir antes de importar torch.
for _v in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "TRITON_CUOBJDUMP_PATH", "TRITON_NVDISASM_PATH"):
    os.environ.pop(_v, None)

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

RAIZ = Path(__file__).resolve().parent.parent
MODELO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
COLUNAS = ("prompt", "completion", "chat_template_kwargs")


def carrega_dados(nome):
    with open(RAIZ / "data" / nome, encoding="utf8") as f:
        return Dataset.from_list([{k: json.loads(l)[k] for k in COLUNAS} for l in f])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=os.environ.get("MODELO", MODELO))
    ap.add_argument("--saida", default=str(RAIZ / "runs" / "lora"))
    ap.add_argument("--max-steps", type=int, default=-1, help="só para ensaio; -1 = épocas do config")
    ap.add_argument("--dados", default="sft_treino.jsonl", help="sft_treino_replay.jsonl no experimento de replay")
    ap.add_argument("--lote", type=int, default=None,
                    help="lote por GPU; a acumulação é ajustada para manter o lote efetivo 16 (ex.: 1 no Qwen 7B)")
    ap.add_argument("--semente", type=int, default=None,
                    help="semente de treino e de ordem dos dados (padrão: config.SEMENTE); para medir variância")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.modelo, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.modelo, torch_dtype=torch.bfloat16, device_map="auto",
                                                 trust_remote_code=True, attn_implementation="eager")
    model.config.use_cache = False               # incompatível com gradient checkpointing

    ajustes = {}
    if a.max_steps > 0:
        ajustes = dict(max_steps=a.max_steps, eval_steps=2, save_steps=2, logging_steps=1)
    if a.semente is not None:
        ajustes.update(seed=a.semente, data_seed=a.semente)
    if a.lote:
        ajustes.update(per_device_train_batch_size=a.lote, per_device_eval_batch_size=a.lote,
                       gradient_accumulation_steps=16 // a.lote)
    args = config.sft(a.saida, **ajustes)
    tr = SFTTrainer(model=model, args=args, processing_class=tok, peft_config=config.lora(),
                    train_dataset=carrega_dados(a.dados),
                    eval_dataset=carrega_dados("sft_validacao.jsonl"))
    tr.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr.train()
    minutos = (time.time() - t0) / 60
    tr.save_model(str(Path(a.saida) / "final"))     # com load_best_model_at_end, é o melhor checkpoint

    hist = {"minutos": round(minutos, 1),
            "memoria_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            "gpu": torch.cuda.get_device_name(0), "melhor_checkpoint": tr.state.best_model_checkpoint,
            "melhor_perda_validacao": tr.state.best_metric, "passos": tr.state.global_step,
            "log": tr.state.log_history}
    Path(a.saida, "historico.json").write_text(json.dumps(hist, indent=1))
    print({k: v for k, v in hist.items() if k != "log"})


if __name__ == "__main__":
    main()
