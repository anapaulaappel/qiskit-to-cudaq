"""Roda em processo próprio: carrega um candidato e salva o vetor de estado. Não conhece a referência."""
import importlib.util, inspect, json, sys
import numpy as np


def main(caminho, saida):
    import cudaq
    from cudaq.kernel.kernel_decorator import PyKernelDecorator
    cudaq.set_target("qpp-cpu")
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
    if inspect.signature(alvo.kernelFunction).parameters:
        return {"status": "exige_argumentos"}    # sem isto, roda com lixo e parece válido
    try:
        psi = np.array(cudaq.get_state(alvo), dtype=np.complex128)
    except BaseException as e:
        tipo = "erro_compilacao" if type(e).__name__ == "CompilerError" else "erro_execucao"
        return {"status": tipo, "erro": f"{type(e).__name__}: {e}"[:300]}
    np.save(saida, psi)
    return {"status": "ok", "n_amplitudes": int(psi.size)}


if __name__ == "__main__":
    print("\n" + json.dumps(main(sys.argv[1], sys.argv[2])), flush=True)
