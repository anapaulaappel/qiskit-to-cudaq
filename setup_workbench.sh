#!/usr/bin/env bash
# Prepara o workbench do OpenShift AI (imagem "Jupyter | PyTorch LLM Compressor | CUDA", Python 3.12,
# 1 GPU L4) com as versões pinadas do projeto. Rodar uma vez, de dentro da pasta do projeto:
#     bash setup_workbench.sh
# Não escreve nada no cluster: só instala pacotes no ambiente do próprio workbench e baixa o modelo.
set -euo pipefail
cd "$(dirname "$0")"

# A imagem aponta o pip para o índice curado da Red Hat AI (packages.redhat.com, CUDA 13), que não
# tem as versões pinadas (ex.: transformers 4.56.2). Os pacotes do projeto vêm do PyPI público.
# Os avisos de conflito com llmcompressor/compressed-tensors da imagem são esperados: não os usamos.
PYPI="--index-url https://pypi.org/simple"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== torch da imagem"
python -c "import torch, sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Versão fixada: torch 2.7.1 com cu128. Só reinstala se a imagem trouxer outra versão.
if ! python -c "import torch; assert torch.__version__.startswith('2.7.1')" 2>/dev/null; then
  pip install -q "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu128
fi

echo "== bibliotecas de treino (versões pinadas)"
pip install -q -U $PYPI "transformers==4.56.2" tokenizers "trl==0.22.2" \
  accelerate datasets peft pandas tqdm huggingface_hub safetensors

echo "== mamba_ssm e causal_conv1d"
# --no-build-isolation é obrigatório. Os dois baixam uma wheel pré-compilada que case com
# torch/CUDA/Python; se não houver, tentam compilar e precisam de nvcc.
# Sem -U e repetindo os pinos: o mamba_ssm depende de transformers sem versão e o pip poderia
# trocar a 4.56.2 pela 5.x. torch==2.7.1 já está satisfeito pela 2.7.1+cu128.
pip install -q $PYPI --no-build-isolation "mamba_ssm==2.2.5" "causal_conv1d==1.5.2" \
  "transformers==4.56.2" "torch==2.7.1"

python - <<'EOF'
import torch, transformers, trl, peft, mamba_ssm, causal_conv1d
print("torch", torch.__version__, "| transformers", transformers.__version__, "| trl", trl.__version__,
      "| peft", peft.__version__, "| mamba_ssm", mamba_ssm.__version__, "| causal_conv1d", causal_conv1d.__version__)
assert transformers.__version__ == "4.56.2" and trl.__version__ == "0.22.2", "versões fora do pino"
EOF

echo "== modelo (7,9 GB, uma vez só, no disco persistente)"
export HF_HOME="${HF_HOME:-$PWD/.hf}"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'))"

echo "== conferência de tokens (treino = geração), no ambiente real"
python train/confere_tokens.py

echo "== tarefas para a geração"
[ -f data/tarefas_modelo.jsonl ] || python eval/gerar.py --preparar

echo "== teste de fumaça: 3 tarefas com o modelo base"
python eval/gerar.py --nome fumaca --limite 3
python - <<'EOF'
import json
for l in open("data/saidas/fumaca.jsonl"):
    s = json.loads(l); print("-----", s["id"], "| truncada:", s["truncadas"][0]); print(s["amostras"][0][:600])
EOF
echo "PRONTO. Próximo passo: python eval/gerar.py --nome baseline"
echo "Lembrete: export HF_HOME=$HF_HOME nas próximas sessões do terminal."
