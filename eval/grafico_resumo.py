"""Gráfico-resumo do README: três painéis (estilo novo, todas as traduções, QuantumKatas), uma barra
por modelo, com intervalo de Wilson de 95%. Lê os números de results/*.json e grava results/resumo.svg.
SVG em Python puro, sem dependências.
    python eval/grafico_resumo.py
"""
import json, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eval"))
from translate_eval import wilson

MODELOS = [("baseline", "Base model"), ("fewshot", "Base + 4-shot prompt"), ("finetuned", "LoRA"),
           ("replay", "LoRA + replay (script)"), ("estilos", "LoRA + style variety (A)"), ("funcao", "LoRA + replay (function, B)")]
PAINEIS = [("teste_estilo", "Unseen Qiskit style (200 tasks)"), ("todas", "All translations (640 tasks)"),
           ("katas", "Qiskit regression: QuantumKatas (349 tasks)")]
COR = {"superficie": "#fcfcfb", "serie": "#2a78d6", "texto": "#0b0b0b", "texto2": "#52514e", "grade": "#e6e5e1"}


def valor(modelo, painel):
    """(taxa, (ic_inf, ic_sup), anotação) ou None."""
    if painel == "katas":
        if modelo == "fewshot":                 # modelo inalterado: mesmo resultado do base
            modelo, nota = "baseline", "unchanged"
        else:
            nota = ""
        r = json.loads((RAIZ / "results" / f"{modelo}_katas.json").read_text())
        return r["taxa"], tuple(r["ic95"]), nota
    r = json.loads((RAIZ / "results" / f"{modelo}_traducao.json").read_text())
    g = r["geral"] if painel == "todas" else r["por_grupo"][painel]
    return g["pass_1"], tuple(g["pass_1_ic95"]), ""


def barra(x0, y, largura, altura, raio=4):
    """Base reta em x0, ponta arredondada (raio 4) no fim."""
    if largura <= 0:
        return ""
    r = min(raio, largura, altura / 2)
    x1 = x0 + largura
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} '
            f'V{y + altura - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + altura:.1f} H{x0:.1f} Z" fill="{COR["serie"]}"/>')


def main():
    esq, topo, larg_painel, gap, linha, esp = 196, 74, 236, 34, 16, 34
    altura = topo + esp * len(MODELOS) + 46
    largura = esq + len(PAINEIS) * larg_painel + (len(PAINEIS) - 1) * gap + 24
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}" '
         f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" role="img" '
         f'aria-label="State-equivalence rates for six models on three test sets">',
         f'<rect width="100%" height="100%" fill="{COR["superficie"]}"/>',
         f'<text x="16" y="28" font-size="16" font-weight="600" fill="{COR["texto"]}">Fine-tuning vs prompting: state equivalence (fidelity ≥ 0.999)</text>',
         f'<text x="16" y="48" font-size="12" fill="{COR["texto2"]}">Same tasks for every model; whiskers are 95% Wilson intervals.</text>']
    for i, (_, nome) in enumerate(MODELOS):
        y = topo + i * esp + linha / 2 + 4
        s.append(f'<text x="{esq - 12}" y="{y:.1f}" font-size="12" text-anchor="end" fill="{COR["texto"]}">{nome}</text>')
    for j, (painel, titulo) in enumerate(PAINEIS):
        x0 = esq + j * (larg_painel + gap)
        escala = (larg_painel - 46) / 100.0
        s.append(f'<text x="{x0}" y="{topo - 12}" font-size="12" font-weight="600" fill="{COR["texto"]}">{titulo}</text>')
        for marca in (0, 50, 100):                          # grade recessiva
            xg = x0 + marca * escala
            s.append(f'<line x1="{xg:.1f}" y1="{topo - 4}" x2="{xg:.1f}" y2="{topo + esp * len(MODELOS) - 10}" stroke="{COR["grade"]}" stroke-width="1"/>')
            s.append(f'<text x="{xg:.1f}" y="{topo + esp * len(MODELOS) + 6}" font-size="11" text-anchor="middle" fill="{COR["texto2"]}">{marca}%</text>')
        taxa_base = valor("baseline", "katas")[0]
        if painel == "katas":                               # referência: modelo base
            xb = x0 + 100 * taxa_base * escala
            s.append(f'<line x1="{xb:.1f}" y1="{topo - 4}" x2="{xb:.1f}" y2="{topo + esp * len(MODELOS) - 10}" stroke="{COR["texto2"]}" stroke-width="1" stroke-dasharray="3 3"/>')
            s.append(f'<text x="{xb + 4:.1f}" y="{topo + esp * len(MODELOS) + 22}" font-size="11" fill="{COR["texto2"]}">base 23.8%</text>')
        for i, (chave, _) in enumerate(MODELOS):
            taxa, (lo, hi), nota = valor(chave, painel)
            y = topo + i * esp
            s.append(barra(x0, y, 100 * taxa * escala, linha))
            yc = y + linha / 2
            s.append(f'<line x1="{x0 + 100 * lo * escala:.1f}" y1="{yc:.1f}" x2="{x0 + 100 * hi * escala:.1f}" y2="{yc:.1f}" stroke="{COR["texto2"]}" stroke-width="1.5"/>')
            rotulo = f"{100 * taxa:.1f}%" + (f" ({nota})" if nota else "")
            fim = max(taxa, hi, taxa_base if painel == "katas" else 0)   # não cruzar a linha de referência
            xt = x0 + 100 * fim * escala + 6
            s.append(f'<text x="{xt:.1f}" y="{yc + 4:.1f}" font-size="11" fill="{COR["texto"]}">{rotulo}</text>')
    s.append("</svg>")
    (RAIZ / "results" / "resumo.svg").write_text("\n".join(s), encoding="utf8")
    print("results/resumo.svg", largura, "x", altura)


if __name__ == "__main__":
    main()
