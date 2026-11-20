"""
GEPA: otimização do prompt de sistema para a tradução Qiskit -> CUDA-Q, sem mexer nos pesos.

Mesmas informações que o LoRA viu: trainset = amostra do sft_treino, valset = amostra do sft_validacao.
O teste (640 tarefas) nunca entra na otimização. Métrica e feedback: eval/pontua.py (verificador
isolado + taxonomia de erros). Depois, o melhor prompt é avaliado no teste com eval/gerar_vllm.py
--prompt-sistema <arquivo> (mesmo pipeline de geração e avaliação dos outros modelos).

Roda no workbench do cluster (precisa do vLLM do Nemotron e do Gemma para a reflexão):
  pip install gepa==0.1.4
  python eval/gepa_otimiza.py --max-metric-calls 300 \
      --task-url http://vllm-nemotron.quantum-finetune.svc:8000 \
      --reflection-url <URL OpenAI-compatible do Gemma> --reflection-model <nome servido>
Saída: results/gepa/melhor_prompt.txt, results/gepa/historico.json

Ensaio local sem GPU nem GEPA (confere dados, prompt e métrica com respostas de referência):
  python eval/gepa_otimiza.py --ensaio
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
from pontua import pontua  # noqa: E402

SEMENTE = 20261117
PROMPT_INICIAL = ("You translate Qiskit circuits into equivalent NVIDIA CUDA-Q kernels. "
                  "Answer with a single Python code block.")


def prompt_com_exemplos(conjunto):
    """Instrução + os 4 exemplos do conjunto do few-shot (mesmos de gerar_vllm.exemplos_fewshot),
    escritos dentro do prompt de sistema, que é o único texto que o GEPA edita."""
    from gerar_vllm import exemplos_fewshot
    msgs, ids = exemplos_fewshot(4, conjunto)
    partes = [PROMPT_INICIAL, "", "Examples:"]
    for u, r in zip(msgs[0::2], msgs[1::2]):
        partes += ["", "Input:", u["content"], "", "Output:", r["content"]]
    return "\n".join(partes), ids


def carrega(nome, n, rng):
    linhas = [json.loads(l) for l in open(RAIZ / "data" / nome, encoding="utf8")]
    rng.shuffle(linhas)
    return linhas[:n]


def enunciado(par):
    return par["prompt"][0]["content"]


def chama_vllm(url, modelo, sistema, usuario, max_tokens=1024, timeout=300):
    corpo = {"model": modelo, "temperature": 0.0, "max_tokens": max_tokens,
             "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
             "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        c = json.loads(r.read())["choices"][0]
    return c["message"]["content"], c.get("finish_reason") == "length"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-treino", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--max-metric-calls", type=int, default=300)
    ap.add_argument("--task-url", default="http://vllm-nemotron.quantum-finetune.svc:8000")
    ap.add_argument("--task-model", default="nemotron")
    ap.add_argument("--reflection-url")
    ap.add_argument("--reflection-model")
    ap.add_argument("--modo", default="graduado", choices=["binario", "graduado"],
                    help="nota usada para guiar a otimização (a avaliação final no teste é sempre binária)")
    ap.add_argument("--reflection-api-key", default="none", help="o endpoint interno não exige chave; o litellm exige uma")
    ap.add_argument("--semente-fewshot", type=int, default=None,
                    help="começa do prompt com os 4 exemplos deste conjunto (ex.: 2, o conjunto fraco)")
    ap.add_argument("--saida-nome", default="gepa", help="pasta em results/ (não sobrescreve outras rodadas)")
    ap.add_argument("--ensaio", action="store_true", help="sem GPU: pontua as respostas de referência")
    a = ap.parse_args()

    rng = random.Random(SEMENTE)
    treino, val = carrega("sft_treino.jsonl", a.n_treino, rng), carrega("sft_validacao.jsonl", a.n_val, rng)
    out = RAIZ / "results" / a.saida_nome
    inicial, ids_exemplos = (prompt_com_exemplos(a.semente_fewshot) if a.semente_fewshot is not None
                             else (PROMPT_INICIAL, []))
    out.mkdir(parents=True, exist_ok=True)

    if a.ensaio:
        t0, ok = time.time(), 0
        for par in val[:10]:
            r = pontua(par, par["completion"][0]["content"])
            ok += r["score"] == 1.0
        print(f"ensaio: {len(treino)} treino, {len(val)} validação; referência pontua {ok}/10 "
              f"({(time.time() - t0) / 10:.1f} s por métrica)")
        return

    import gepa
    from gepa.adapters.default_adapter.default_adapter import EvaluationResult
    # gepa 0.1.4 (conferido no código-fonte): dados {"input", "answer", "additional_context"};
    # task_lm(messages) -> str, com messages = [system = candidato, user = data["input"]];
    # evaluator(data, response) -> EvaluationResult(score, feedback, objective_scores).
    por_enunciado = {}

    def para_gepa(pares):
        dados = []
        for par in pares:
            por_enunciado[enunciado(par)] = par
            dados.append({"input": enunciado(par), "answer": par["completion"][0]["content"],
                          "additional_context": {}})
        return dados

    def task_lm(messages):
        sistema = next(m["content"] for m in messages if m["role"] == "system")
        usuario = next(m["content"] for m in messages if m["role"] == "user")
        texto, truncada = chama_vllm(a.task_url, a.task_model, sistema, usuario)
        return texto if not truncada else texto + "\n[TRUNCATED]"

    def avalia(data, resposta):
        par = por_enunciado[data["input"]]
        truncada = resposta.endswith("[TRUNCATED]")
        r = pontua(par, resposta.replace("\n[TRUNCATED]", ""), modo=a.modo, truncada=truncada)
        return EvaluationResult(score=r["score"], feedback=r["feedback"], objective_scores=None)

    resultado = gepa.optimize(
        seed_candidate={"system_prompt": inicial},
        trainset=para_gepa(treino), valset=para_gepa(val),
        task_lm=task_lm, evaluator=avalia,
        reflection_lm=f"openai/{a.reflection_model}",
        reflection_lm_kwargs={"api_base": a.reflection_url, "api_key": a.reflection_api_key},
        max_metric_calls=a.max_metric_calls, reflection_minibatch_size=3, use_merge=True,
        seed=SEMENTE, run_dir=str(out / "run"),
    )
    melhor = resultado.best_candidate["system_prompt"]
    (out / "melhor_prompt.txt").write_text(melhor, encoding="utf8")
    (out / "prompt_inicial.txt").write_text(inicial, encoding="utf8")
    (out / "historico.json").write_text(json.dumps(
        {"prompt_inicial": inicial, "exemplos_iniciais": ids_exemplos, "melhor": melhor, "args": vars(a),
         "val_scores": getattr(resultado, "val_aggregate_scores", None)}, indent=1, ensure_ascii=False))
    print("melhor prompt salvo em", out / "melhor_prompt.txt")


if __name__ == "__main__":
    main()
