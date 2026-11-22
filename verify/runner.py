"""Execução isolada de um kernel candidato: subprocesso, diretório temporário, sem a referência.

O subprocesso é vigiado: se a memória (somando os filhos) passar de LIMITE_MEM_MB, ou o tempo passar do limite,
a árvore inteira é morta e o status é "memoria" ou "timeout". No macOS o RLIMIT_AS não é aplicado; a medida é o
phys_footprint (inclui memória comprimida, que o RSS não conta). Motivo: no CUDA-Q 0.16 (qpp-cpu), uma porta com
controle igual ao alvo, p.ex. x.ctrl(q[0], q[0]), consome memória sem limite na simulação."""
import ctypes, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import psutil
from sanitize import sanitiza

WORKER = Path(__file__).resolve().parent / "_worker.py"
LIMITE_MEM_MB = 3072
INTERVALO = 0.1


class _RUsage(ctypes.Structure):                  # rusage_info_v2 (sys/resource.h), só o começo nos interessa
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f, ctypes.c_uint64) for f in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint")] + [("resto", ctypes.c_uint64 * 40)]


try:
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None
except OSError:
    _libproc = None


def _bytes(p):
    if _libproc is not None:
        r = _RUsage()
        if _libproc.proc_pid_rusage(p.pid, 2, ctypes.byref(r)) == 0:
            return max(r.phys_footprint, r.resident_size)
    return p.memory_info().rss


def _mem_mb(proc):
    total = 0
    try:
        for p in [proc, *proc.children(recursive=True)]:
            try:
                total += _bytes(p)
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total / 2**20


def _mata(proc):
    try:
        procs = [*proc.children(recursive=True), proc]
    except psutil.Error:
        procs = [proc]
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=5)


def _roda(cmd, cwd, timeout, limite_mb=None):
    """Como subprocess.run(capture_output=True, text=True), vigiado. Devolve (stdout, stderr, motivo); motivo é
    None, "timeout" ou "memoria"."""
    limite_mb = limite_mb or LIMITE_MEM_MB
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, text=True)
        proc, fim, motivo = psutil.Process(p.pid), time.monotonic() + timeout, None
        while p.poll() is None:
            if _mem_mb(proc) > limite_mb:
                motivo = "memoria"
            elif time.monotonic() > fim:
                motivo = "timeout"
            if motivo:
                _mata(proc); p.wait()
                break
            time.sleep(INTERVALO)
        out.seek(0); err.seek(0)
        return out.read(), err.read(), motivo


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
        stdout, stderr, motivo = _roda([sys.executable, "-I", "-W", "ignore", str(WORKER), str(arq), str(saida)],
                                       tmp, timeout)
        if motivo:
            return {"status": motivo}, None
        try:
            res = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return {"status": "falha_processo", "erro": stderr[-300:]}, None
        return res, (np.load(saida) if res["status"] == "ok" else None)


WORKER_ARGS = Path(__file__).resolve().parent / "_worker_args.py"


def executa_args(fonte, conjuntos, timeout=120):
    """Conjunto converter-hard: kernel com argumentos, chamado com cada conjunto. Devolve (resultado, [vetores])."""
    for _ in range(2):
        try:
            limpo = sanitiza(fonte)
        except (SyntaxError, ValueError) as e:
            return {"status": "erro_sintaxe", "erro": str(e)[:300]}, None
        with tempfile.TemporaryDirectory() as tmp:
            arq, saida = Path(tmp) / "candidato.py", Path(tmp) / "estados.npz"
            arq.write_text(limpo, encoding="utf8")
            stdout, _, motivo = _roda([sys.executable, "-I", "-W", "ignore", str(WORKER_ARGS), str(arq), str(saida),
                                       json.dumps(conjuntos)], tmp, timeout)
            if motivo:
                return {"status": motivo}, None
            try:
                res = json.loads(stdout.strip().splitlines()[-1])
            except (IndexError, ValueError):
                continue                                    # falha de ambiente: repete uma vez
            if res["status"] != "ok":
                return res, None
            z = np.load(saida)
            return res, [z[f"arr_{i}"] for i in range(len(conjuntos))]
    return {"status": "falha_processo"}, None
