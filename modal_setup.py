"""
Setup do Modal para a trilha quântica — Nemotron 3 Nano 4B.

Uso, na ordem:

    pip install modal
    modal setup                                  # autentica
    modal run modal_setup.py::build_smoke        # constrói a imagem e testa geração
    modal run modal_setup.py::download_model     # baixa os pesos para o Volume (uma vez só)

Depois disso a imagem fica em cache e os pesos ficam no Volume.
Nenhuma das duas coisas precisa ser refeita.

Atenção a duas coisas que quebram silenciosamente:
  1. O timeout padrão do Modal é 300 s. Treino SEM `timeout=` morre no meio.
  2. GPU exige forma de pagamento cadastrada, mesmo usando o crédito gratuito.
     Cadastre hoje, não no dia do treino.
"""

import modal

# ---------------------------------------------------------------- imagem

TORCH_IDX = "https://download.pytorch.org/whl/cu128"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .run_commands(
        f"pip install 'torch==2.7.1' 'torchvision==0.22.1' 'torchaudio==2.7.1' "
        f"--index-url {TORCH_IDX}",
        "pip install -U 'transformers==4.56.2' tokenizers 'trl==0.22.2' "
        "accelerate datasets peft pandas tqdm huggingface_hub safetensors",
        # Esta camada é a frágil: mamba_ssm e causal_conv1d compilam kernels CUDA.
        # O gpu= aqui existe para que a compilação enxergue uma GPU.
        "pip install -U --no-build-isolation 'mamba_ssm==2.2.5' 'causal_conv1d==1.5.2'",
        gpu="L4",
    )
)
# SE A ÚLTIMA CAMADA FALHAR por falta de nvcc:
#   opção A — instale a wheel pré-compilada do mamba_ssm (releases do GitHub do
#             projeto, escolhendo a que casa com torch 2.7 / cu128 / cp312).
#             Evita compilar e é o que a maioria das pessoas acaba fazendo.
#   opção B — troque a base por uma imagem CUDA devel (que traz o nvcc), via
#             modal.Image.from_registry(...). Veja modal.com/docs/guide/images.

app = modal.App("nemotron-quantum", image=image)

# ---------------------------------------------------------------- volume

vol = modal.Volume.from_name("nemotron-vol", create_if_missing=True, version=2)
VOL = "/vol"

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MODEL_DIR = f"{VOL}/models/{MODEL_ID}"


# ------------------------------------------------- baixar pesos (uma vez)

@app.function(volumes={VOL: vol}, timeout=60 * 60)
def download_model():
    import os
    from huggingface_hub import snapshot_download

    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"já existe em {MODEL_DIR}, pulando")
        return

    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
    vol.commit()
    print(f"baixado para {MODEL_DIR}")


# ------------------------------------------------------- teste de fumaça

@app.function(gpu="L4", volumes={VOL: vol}, timeout=30 * 60)
def build_smoke():
    """Confirma que o stack subiu e que o modelo gera. Rode isto primeiro."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_ID

    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",   # obrigatório neste modelo
    )

    msgs = [{
        "role": "user",
        "content": (
            "Traduza este circuito Qiskit para um kernel CUDA-Q equivalente:\n\n"
            "qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\n"
        ),
    }]
    ids = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,         # raciocínio vem LIGADO por padrão
    ).to(model.device)

    out = model.generate(ids, max_new_tokens=256, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))


# --------------------------------------------------- esqueleto do treino

@app.function(gpu="L4", volumes={VOL: vol}, timeout=6 * 60 * 60)   # 6h, NÃO 300s
def train():
    """
    Preencher no Q4. O que importa já está aqui:
      - timeout generoso
      - output_dir dentro do Volume, para o adaptador sobreviver
      - vol.commit() no fim
    """
    import os, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    out_dir = f"{VOL}/runs/lora-qiskit2cudaq"
    os.makedirs(out_dir, exist_ok=True)

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.1,
        bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=4,     # suba para 8 se couber
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        num_train_epochs=2,
        max_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        completion_only_loss=True,         # perda só na resposta
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
    )

    # ... carregar dataset, montar o SFTTrainer, treinar ...

    vol.commit()


# ------------------------------------------------------------------------
# Para depurar dentro do mesmo ambiente, sem queimar GPU num script inteiro:
#     modal shell modal_setup.py::build_smoke
# Para baixar o adaptador depois:
#     modal volume get nemotron-vol runs/lora-qiskit2cudaq ./local
