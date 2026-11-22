"""Como _worker.py, para kernels COM argumentos (conjunto converter-hard): chama o kernel com cada conjunto
de argumentos declarado na tarefa e salva os vetores de estado. Não conhece a referência."""
import importlib.util, inspect, json, sys
import numpy as np


def main(caminho, saida, args_json):
    import cudaq
    from cudaq.kernel.kernel_decorator import PyKernelDecorator
    cudaq.set_target("qpp-cpu")
    conjuntos = json.loads(args_json)
    spec = importlib.util.spec_from_file_location("candidato", caminho)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException as e:
        return {"status": "erro_compilacao", "erro": f"{type(e).__name__}: {e}"[:300]}
    kernels = {k: v for k, v in vars(mod).items() if isinstance(v, PyKernelDecorator)}
    if not kernels:
        return {"status": "sem_kernel"}
    alvo = kernels.get("kernel") or list(kernels.values())[-1]
    if len(inspect.signature(alvo.kernelFunction).parameters) != len(conjuntos[0]):
        return {"status": "assinatura_errada"}
    estados = []
    for args in conjuntos:
        try:
            estados.append(np.array(cudaq.get_state(alvo, *args), dtype=np.complex128))
        except BaseException as e:
            tipo = "erro_compilacao" if type(e).__name__ == "CompilerError" else "erro_execucao"
            return {"status": tipo, "erro": f"{type(e).__name__}: {e}"[:300]}
    np.savez(saida, *estados)
    return {"status": "ok", "n": len(estados)}


if __name__ == "__main__":
    print("\n" + json.dumps(main(sys.argv[1], sys.argv[2], sys.argv[3])), flush=True)
