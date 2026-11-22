"""Conversor de circuito no conjunto converter-hard, com a melhor chance possível: instancia a função Qiskit
com o PRIMEIRO conjunto de argumentos da tarefa, transpila e gera um kernel (sem parâmetros, como qualquer
conversor que parte de um circuito executado). Saída: data/saidas/hard_conversor.jsonl"""
import json, sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
src = (RAIZ / "eval" / "linha_base_circuito.py").read_text().split("T = carrega_tarefas()")[0]
exec(src)
out = []
for l in open(RAIZ / "data" / "teste_converter_hard.jsonl"):
    t = json.loads(l)
    codigo = t["qiskit"] + f"\nqc = {t['funcao']}(*{t['args'][0]!r})\n"
    out.append({"id": t["id"], "amostras": ["```python\n" + traduz(codigo) + "```"]})
(RAIZ / "data" / "saidas" / "hard_conversor.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
