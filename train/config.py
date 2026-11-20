"""Configuração do LoRA e do treino, num lugar só. Usada pelo treino na Modal e pelo ensaio na CPU.

Mudanças em relação à configuração LoRA de partida, e o motivo de cada uma:
- Lote efetivo 16 (4 x 4) em vez de 64 (4 x 16). Com 1.355 exemplos e 2 épocas, 64 daria cerca de
  42 atualizações de pesos; 16 dá cerca de 170.
- learning_rate 1e-4 em vez de 5e-5, com aquecimento de 5% e decaimento cosseno. LoRA costuma usar
  de 1e-4 a 2e-4; 5e-5 com poucas atualizações tende a subtreinar. A curva de validação mostra se
  passou do ponto.
- Avaliação na validação a cada 20 passos, e o adaptador final é o de menor perda de validação
  (load_best_model_at_end). Os conjuntos de teste não participam dessa escolha.
- LoRA só na atenção (q, k, v, o) e no MLP (up, down), não em "all-linear". O PEFT 0.21 recusa
  LoRA em out_proj e conv1d do Mamba neste modelo (ValueError no ensaio na GPU, 05/11/2026). E no
  vLLM a in_proj do Mamba é uma camada fundida de 5 partes, enquanto o mapeamento de LoRA do
  Nemotron-H só cobre qkv: um adaptador nela provavelmente não carregaria no servidor. As 21
  camadas Mamba ficam congeladas; atenção (4) e MLP (17) recebem o adaptador.
O resto é igual à configuração de partida: r=32, alpha=64, dropout 0,1, BF16, max_length 1024,
completion_only_loss, gradient checkpointing, sem packing, adamw_torch_fused.
"""
from peft import LoraConfig
from trl import SFTConfig

SEMENTE = 20261105
PASSOS_AVALIACAO = 20
MODULOS_LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def lora():
    return LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, bias="none",
                      task_type="CAUSAL_LM", target_modules=MODULOS_LORA)


def sft(output_dir, **ajustes):
    """`ajustes` existe só para o ensaio na CPU (sem BF16, poucos passos). Na Modal, não passar nada."""
    base = dict(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,          # lote efetivo 16
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=2,
        max_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        completion_only_loss=True,              # perda só na resposta
        optim="adamw_torch_fused",
        eval_strategy="steps",
        eval_steps=PASSOS_AVALIACAO,
        save_strategy="steps",                  # precisa casar com a avaliação para escolher o melhor
        save_steps=PASSOS_AVALIACAO,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        seed=SEMENTE,
        data_seed=SEMENTE,
        report_to=[],
    )
    return SFTConfig(**(base | ajustes))
