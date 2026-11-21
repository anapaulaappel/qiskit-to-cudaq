"""Sensibilidade ao limiar de fidelidade (offline): pass@1 com 0,99 / 0,999 / 0,9999 a partir das
fidelidades já gravadas em results/<modelo>_traducao.json. Saída: results/limiar_fidelidade.json"""
import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"
MODELOS = ["fewshot", "finetuned", "seed1", "seed2", "replay", "estilos", "funcao", "grpo", "qwen_fewshot"]
LIMIARES = [0.99, 0.999, 0.9999]
out = {}
for m in MODELOS:
    ts = json.load(open(R / f"{m}_traducao.json"))["por_tarefa"]
    out[m] = {}
    for g in (None, "teste_estilo"):
        sub = [t for t in ts if g is None or t["divisao"] == g]
        f = [t["amostras"][0].get("fidelidade") or 0.0 for t in sub]
        out[m][g or "todas"] = {str(l): round(100 * sum(x >= l for x in f) / len(f), 1) for l in LIMIARES}
(R / "limiar_fidelidade.json").write_text(json.dumps(out, indent=1))
for m, v in out.items(): print(f"{m:13s}", v["todas"], v["teste_estilo"])
