"""
Piloto de LoRA + GRPO: o RL com recompensa verificável generaliza melhor que o SFT para estilos novos?

Desenho (comparação limpa com o experimento A):
  - ponto de partida: o LoRA do SFT original (runs/lora/final, estilo canônico), continuado com RL;
  - prompts: os pares de data/sft_treino_estilos.jsonl cujo enunciado foi reescrito com estilos
    (os mesmos dados que o A usou no SFT; os idiomas `append` e `funcao` ficam de fora, como no A);
  - recompensa: eval/pontua.py, modo graduado (1 equivalente; 0,1 + 0,4·fidelidade se roda; 0 se não);
  - avaliação depois: as 200 tarefas do teste de estilo, com atenção aos idiomas reservados
    (A, SFT com estilos: 21,7% nos reservados).

Roda na venv-grpo (TRL 1.18.1, vLLM 0.33, transformers 5.20, Nemotron-H nativo), 1 GPU L4, vLLM em
modo colocate (o generate do transformers é lento neste modelo; os rollouts vêm do vLLM).
  python train/grpo_piloto.py --max-steps 30 --saida runs/grpo-piloto
"""
import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
MODELO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def prompts_com_estilo():
    canon = {json.loads(l)["id"]: json.loads(l) for l in open(RAIZ / "data" / "sft_treino.jsonl", encoding="utf8")}
    out = []
    for l in open(RAIZ / "data" / "sft_treino_estilos.jsonl", encoding="utf8"):
        p = json.loads(l)
        c = canon.get(p["id"])
        if c is None or p["prompt"][0]["content"] != c["prompt"][0]["content"]:   # enunciado reescrito
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--saida", default=str(RAIZ / "runs" / "grpo-piloto"))
    ap.add_argument("--adaptador", default=str(RAIZ / "runs" / "lora" / "final"))
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--geracoes", type=int, default=4)
    ap.add_argument("--vllm-mem", type=float, default=0.45)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from pontua import pontua

    pares = prompts_com_estilo()
    print(f"{len(pares)} prompts com estilo")
    ds = Dataset.from_list([{"prompt": p["prompt"], "idx": i} for i, p in enumerate(pares)])

    pool = cf.ThreadPoolExecutor(8)

    def recompensa(prompts, completions, idx, **kw):
        textos = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        futuros = [pool.submit(pontua, pares[i], t, "graduado") for i, t in zip(idx, textos)]
        return [f.result()["score"] for f in futuros]

    tok = AutoTokenizer.from_pretrained(MODELO)
    base = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16)
    modelo = PeftModel.from_pretrained(base, a.adaptador, is_trainable=True)

    cfg = GRPOConfig(
        output_dir=a.saida, max_steps=a.max_steps, learning_rate=a.lr,
        per_device_train_batch_size=a.geracoes, gradient_accumulation_steps=2,
        num_generations=a.geracoes, max_completion_length=512,
        bf16=True, gradient_checkpointing=True, beta=0.0,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=2048,
        chat_template_kwargs={"enable_thinking": False},
        mask_truncated_completions=True, log_completions=False,
        logging_steps=1, save_steps=max(10, a.max_steps // 3), report_to=[], seed=20261119,
    )
    tr = GRPOTrainer(model=modelo, reward_funcs=recompensa, args=cfg, train_dataset=ds, processing_class=tok)
    tr.train()
    tr.save_model(str(Path(a.saida) / "final"))
    rec = [h.get("reward") for h in tr.state.log_history if "reward" in h]
    (Path(a.saida) / "historico.json").write_text(json.dumps(
        {"args": vars(a), "n_prompts": len(pares), "reward_por_passo": rec,
         "memoria_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
         "log": tr.state.log_history}, indent=1))
    print("recompensa por passo:", [round(r, 3) for r in rec if r is not None])


if __name__ == "__main__":
    main()
