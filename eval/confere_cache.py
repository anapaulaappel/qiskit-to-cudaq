"""Confere que a geração com o cache híbrido (gerar.cache_hibrido) dá a mesma saída que sem cache.

Roda no workbench, antes da linha de base:
    python eval/confere_cache.py            # 6 tarefas, até 256 tokens novos cada

Para cada tarefa, gera greedy duas vezes com o mesmo modelo carregado: com a função original do
modelo (sem cache efetivo) e com o cache ligado. Compara os tokens gerados e o tempo. Em BF16, o
caminho com cache faz as contas em outra ordem, então uma divergência tardia e rara é possível; o
que importa é não haver divergência sistemática. Sai com erro se mais de 1 tarefa divergir.
"""
import json, os, sys, time
from pathlib import Path

for _v in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "TRITON_CUOBJDUMP_PATH", "TRITON_NVDISASM_PATH"):
    os.environ.pop(_v, None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gerar
import torch

N, NOVOS = 6, 256


def gera(model, tok, ids, fim):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=NOVOS, do_sample=False,
                           pad_token_id=tok.eos_token_id, eos_token_id=fim)[0]
    torch.cuda.synchronize()
    return g[ids.shape[-1]:].tolist(), time.time() - t


def main():
    with open(gerar.TAREFAS, encoding="utf8") as f:
        tarefas = [json.loads(l) for l in f]
    # algumas traduções de cada conjunto e um Kata
    escolha = [t for t in tarefas if t["tipo"] == "traducao"][::120][:N - 1] + [t for t in tarefas if t["tipo"] == "kata"][:1]
    tok, model = gerar.carrega(os.environ.get("MODELO", gerar.MODELO), "", usa_cache=False)
    fim = gerar.ids_de_parada(tok, model)
    original = model.prepare_inputs_for_generation

    divergem, t_sem, t_com = 0, 0.0, 0.0
    for t in escolha:
        ids = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], add_generation_prompt=True,
                                      return_tensors="pt", enable_thinking=False).to(model.device)
        model.prepare_inputs_for_generation = original
        a, ta = gera(model, tok, ids, fim)
        gerar.cache_hibrido(model)
        b, tb = gera(model, tok, ids, fim)
        t_sem += ta; t_com += tb
        igual = a == b
        primeiro = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
        divergem += not igual
        print(f"{t['id']:24s} {t['tipo']:9s} tokens {len(a):4d}/{len(b):4d}  sem cache {ta:6.1f}s  com cache {tb:5.1f}s  "
              f"{'IGUAL' if igual else f'DIVERGE no token {primeiro}'}", flush=True)
    print(f"\n{len(escolha) - divergem}/{len(escolha)} iguais | tempo total sem cache {t_sem:.0f}s, com cache {t_com:.0f}s "
          f"({t_sem / max(t_com, 1e-9):.1f}x)")
    sys.exit(1 if divergem > 1 else 0)


if __name__ == "__main__":
    main()
