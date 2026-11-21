"""
Taxonomia dos erros de tradução, por análise estática (o código do modelo NÃO é executado).

Para cada tarefa que falhou na primeira amostra (results/<modelo>_traducao.json), classifica o
código gerado (data/saidas/<modelo>.jsonl) numa categoria, inspirada nas categorias de bugs de
tradução de Pan et al. (ICSE 2024), adaptadas a Qiskit → CUDA-Q:

  sem_codigo_cudaq    resposta sem kernel CUDA-Q (só comentários, texto, CUDA C)
  cuda_c              confunde com CUDA C (__global__, void kernel, #include)
  api_qiskit          usa API do Qiskit dentro do kernel (qc., QuantumCircuit, .cx/.append)
  idioma_qiskit       copia uma construção do Qiskit sem equivalente direto (barrier, measure_all)
  classe_qiskit       usa nome de classe de porta do Qiskit (X, H, CRY, Swap; vem do idioma append)
  nome_porta_qiskit   usa nome de porta do Qiskit que não existe no CUDA-Q (cx, cz, cp, cry, p, sx)
  porta_inventada     chama porta que não existe em nenhum dos dois (r_y, cnot, ...)
  indice_cru          usa índices inteiros sem alocar cudaq.qvector (rx(0.3, 0))
  qubits_errados      aloca um número de qubits diferente da referência
  kernel_argumentos   kernel com argumentos (o enunciado pede sem argumentos)
  sintaxe             outro erro de sintaxe Python
  truncada            saída cortada pelo limite de tokens
  -- o código roda, mas o estado é outro (comparação com a referência CUDA-Q):
  porta_faltando      menos operações que a referência
  porta_sobrando      mais operações que a referência
  angulo_errado       mesmas portas, algum ângulo diferente (sinal, valor)
  controle_alvo       mesma multiset de portas, controle e alvo trocados
  ordem_ou_qubit      mesmas portas e ângulos, qubits ou ordem diferentes
  outro_compilacao / outro_estado   não classificados

Uso:  python eval/taxonomia.py            (todos os modelos)
Saída: results/taxonomia_erros.json e uma tabela no terminal
"""
import json
import re
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MODELOS = ["baseline", "fewshot", "finetuned", "seed1", "seed2", "grpo", "qwen_fewshot", "misto", "replay", "estilos", "funcao"]
ARQUIVOS = ["teste_tamanho", "teste_estrutura", "teste_estilo"]
BLOCO = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
PORTAS_CUDAQ = {"x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "r1", "swap", "u3", "exp_pauli",
                "mz", "mx", "my", "reset", "sdg", "tdg"}
CHAMADA = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)(\.ctrl|\.adj)?\s*\(", re.M)
IDIOMA_QISKIT = {"barrier", "measure_all", "measure", "append", "compose", "initialize"}
PORTAS_QISKIT = {"cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "ccx", "cswap", "mcx", "p",
                 "u", "sx", "sxdg", "id", "rxx", "ryy", "rzz", "cu", "u1", "u2", "iswap", "ecr"}
OP = re.compile(r"^\s*([a-z0-9_]+)(\.ctrl|\.adj)?\s*\((.*)\)\s*$", re.M)


def extrai(texto):
    blocos = BLOCO.findall(texto)
    return next((b for b in blocos if "cudaq" in b), blocos[0] if blocos else texto)


def ops(codigo):
    """Lista de (porta, ctrl, ângulos, qubits) das linhas de operação de um kernel CUDA-Q."""
    out = []
    for m in OP.finditer(codigo):
        nome, mod, args = m.group(1), m.group(2) or "", m.group(3)
        if nome not in PORTAS_CUDAQ or nome == "mz":
            continue
        partes = [a.strip() for a in args.split(",")]
        ang = tuple(round(float(a), 4) for a in partes if re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", a))
        qub = tuple(a for a in partes if "q" in a or "[" in a)
        out.append((nome, mod, ang, qub))
    return out


def classifica(texto, truncada, status, ref_cudaq, n_ref):
    cod = extrai(texto)
    if truncada:
        return "truncada"
    if "cudaq" not in cod or "@cudaq.kernel" not in cod:
        if re.search(r"__global__|#include|void\s+\w+\s*\(", texto):
            return "cuda_c"
        return "sem_codigo_cudaq"
    if re.search(r"__global__|#include", cod):
        return "cuda_c"
    if re.search(r"def\s+kernel\s*\(\s*[^)\s]", cod):
        return "kernel_argumentos"
    if re.search(r"\bqc\.|QuantumCircuit|\.append\(|from qiskit", cod):
        return "api_qiskit"
    chamadas = {m.group(1) for m in CHAMADA.finditer(cod)}
    desconhecidas = chamadas - PORTAS_CUDAQ - {"kernel", "print", "range", "len", "int", "float",
                                               "def", "cudaq", "list", "enumerate", "for"}
    classes = {c for c in desconhecidas if c[0].isupper()}
    desconhecidas -= classes
    if status == "erro_sintaxe":
        return "sintaxe"
    m = re.search(r"cudaq\.qvector\(\s*(\d+)\s*\)", cod)
    if m is None and re.search(r"^\s*[a-z0-9_]+(\.ctrl)?\(\s*-?[\d.]+\s*,\s*\d+\s*\)", cod, re.M) \
            or (m is None and re.search(r"^\s*[a-z]+(\.ctrl)?\(\s*\d+\s*(,\s*\d+)?\s*\)", cod, re.M)):
        return "indice_cru"
    if desconhecidas & IDIOMA_QISKIT:
        return "idioma_qiskit"
    if classes:
        return "classe_qiskit"
    if desconhecidas & PORTAS_QISKIT:
        return "nome_porta_qiskit"
    if desconhecidas:
        return "porta_inventada"
    if m and int(m.group(1)) != n_ref:
        return "qubits_errados"
    if status != "ok":
        return "outro_compilacao"
    a, b = ops(cod), ops(ref_cudaq)
    if len(a) < len(b):
        return "porta_faltando"
    if len(a) > len(b):
        return "porta_sobrando"
    if Counter((g, c) for g, c, _, _ in a) == Counter((g, c) for g, c, _, _ in b):
        if [x[2] for x in a] != [x[2] for x in b] and Counter(x[2] for x in a) != Counter(x[2] for x in b):
            return "angulo_errado"
        sa = Counter((g, c, ang, tuple(sorted(q))) for g, c, ang, q in a)
        sb = Counter((g, c, ang, tuple(sorted(q))) for g, c, ang, q in b)
        if sa == sb:
            return "controle_alvo"
        return "ordem_ou_qubit"
    return "outro_estado"


def main():
    tarefas = {}
    for nome in ARQUIVOS:
        for linha in open(RAIZ / "data" / f"{nome}.jsonl", encoding="utf8"):
            t = json.loads(linha)
            t["divisao"] = nome
            tarefas[t["id"]] = t
    resumo = {}
    for mod in MODELOS:
        res = {t["id"]: t for t in json.load(open(RAIZ / "results" / f"{mod}_traducao.json"))["por_tarefa"]}
        saidas = {}
        for linha in open(RAIZ / "data" / "saidas" / f"{mod}.jsonl", encoding="utf8"):
            s = json.loads(linha)
            if s.get("tipo") == "traducao":
                saidas[s["id"]] = s
        cats, exemplos = Counter(), {}
        por_div = {}
        for tid, r in res.items():
            a = r["amostras"][0]
            if a.get("equivalente"):
                continue
            s, t = saidas[tid], tarefas[tid]
            cat = classifica(s["amostras"][0], s["truncadas"][0], a["status"], t["cudaq"], t["n"])
            cats[cat] += 1
            por_div.setdefault(r["divisao"], Counter())[cat] += 1
            exemplos.setdefault(cat, tid)
        resumo[mod] = {"falhas": sum(cats.values()), "categorias": dict(cats.most_common()),
                       "por_divisao": {k: dict(v.most_common()) for k, v in por_div.items()},
                       "exemplo_por_categoria": exemplos}
    (RAIZ / "results" / "taxonomia_erros.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False))
    todas = sorted({c for v in resumo.values() for c in v["categorias"]},
                   key=lambda c: -sum(v["categorias"].get(c, 0) for v in resumo.values()))
    print(f"{'categoria':20s}" + "".join(f"{m:>10s}" for m in MODELOS))
    for c in todas:
        print(f"{c:20s}" + "".join(f"{resumo[m]['categorias'].get(c, 0):>10d}" for m in MODELOS))
    print(f"{'TOTAL falhas':20s}" + "".join(f"{resumo[m]['falhas']:>10d}" for m in MODELOS))


if __name__ == "__main__":
    main()
