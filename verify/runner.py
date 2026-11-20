"""Execução isolada de um kernel candidato: subprocesso, diretório temporário, sem a referência."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from sanitize import sanitiza

WORKER = Path(__file__).resolve().parent / "_worker.py"


def executa(fonte, timeout=60):
    """Repete uma vez se o próprio processo morrer: isso é ambiente, não candidato."""
    res, psi = _executa(fonte, timeout)
    return _executa(fonte, timeout) if res["status"] == "falha_processo" else (res, psi)


def _executa(fonte, timeout):
    """Devolve (resultado, vetor). resultado["status"]: ok, erro_sintaxe, erro_compilacao,
    erro_execucao, sem_kernel, exige_argumentos, timeout ou falha_processo. O vetor só existe com status ok."""
    try:
        limpo = sanitiza(fonte)
    except (SyntaxError, ValueError) as e:
        return {"status": "erro_sintaxe", "erro": str(e)[:300]}, None
    with tempfile.TemporaryDirectory() as tmp:
        arq, saida = Path(tmp) / "candidato.py", Path(tmp) / "estado.npy"
        arq.write_text(limpo, encoding="utf8")
        try:
            p = subprocess.run([sys.executable, "-I", "-W", "ignore", str(WORKER), str(arq), str(saida)],
                               cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}, None
        try:
            res = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return {"status": "falha_processo", "erro": p.stderr[-300:]}, None
        return res, (np.load(saida) if res["status"] == "ok" else None)
