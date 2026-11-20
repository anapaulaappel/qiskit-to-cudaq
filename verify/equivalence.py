"""Fidelidade de estado entre Qiskit e CUDA-Q, com correção de ordenação de qubits."""
import numpy as np


def fidelidade(psi_a, psi_b):
    """|<a|b>|^2, que ignora fase global."""
    if np.size(psi_a) != np.size(psi_b):      # número de qubits diferente: não é equivalente
        return 0.0
    return float(abs(np.vdot(psi_a, psi_b)) ** 2)


def equivalente(psi_a, psi_b, limiar=0.999):
    return fidelidade(psi_a, psi_b) >= limiar


def inverte_bits(psi):
    """Troca entre as convenções: cada índice de n bits é lido de trás para frente."""
    psi = np.asarray(psi)
    n = int(np.log2(psi.size))
    if 2 ** n != psi.size:
        raise ValueError(f"tamanho {psi.size} não é potência de 2")
    return psi.reshape([2] * n).transpose(*reversed(range(n))).reshape(-1)
