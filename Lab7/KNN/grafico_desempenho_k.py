"""Desenha os resultados IBk do Weka, sem voltar a treinar modelos.

Mantém resultados.json na mesma pasta. Abre este ficheiro no editor Python.
Instala: python3 -m pip install matplotlib
Executa: python3 grafico_desempenho_k.py
Janelas: python3 grafico_desempenho_k.py --show
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pasta = Path(__file__).resolve().parent
    dados = json.loads((pasta / "resultados.json").read_text(encoding="utf-8"))
    rows = sorted(dados["evaluations"], key=lambda r: r["k"])
    ks = [r["k"] for r in rows]
    assert ks == [1, 3, 5, 7, 9, 11, 15, 19, 25, 31]
    assert dados["folds"] == 10 and dados["seed"] == 1 and not dados["holdout_used"]
    accuracy, f1, recall = [], [], []
    for r in rows:
        report = pasta / r["source_file"]
        if report.exists():
            assert hashlib.sha256(report.read_bytes()).hexdigest() == r["source_file_sha256"]
        m = r["confusion_matrix_M_B"]
        tp, fn = m["M"]
        fp, tn = m["B"]
        assert tp + fn == 191 and fp + tn == 321
        a, f, rec = 100 * (tp + tn) / 512, 2 * tp / (2 * tp + fp + fn), tp / 191
        assert abs(a - r["accuracy_percent"]) < 0.000051
        assert abs(f - r["exact_metrics_M"]["F1"]) < 1e-12
        assert abs(rec - r["exact_metrics_M"]["recall"]) < 1e-12
        accuracy.append(a)
        f1.append(f)
        recall.append(rec)
    assert ks[accuracy.index(max(accuracy))] == ks[f1.index(max(f1))] == 9
    assert ks[recall.index(max(recall))] == 7
    assert any(b > a for a, b in zip(accuracy, accuracy[1:]))
    assert any(b < a for a, b in zip(accuracy, accuracy[1:]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": "white", "figure.facecolor": "#f8fafc",
                         "savefig.facecolor": "#f8fafc"})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.80, bottom=0.18, hspace=0.50)
    fig.suptitle("KNN: como o desempenho varia com k", x=0.10, y=0.965,
                 ha="left", fontsize=23, weight="bold", color="#16324f")
    fig.text(0.10, 0.915, "IBk do Weka • 512 exemplos de treino • 10 folds • seed 1 • classe positiva M",
             fontsize=11, color="#475569")
    fig.text(0.10, 0.867, "As curvas sobem e descem: a relação observada não é monotónica.",
             fontsize=13, weight="bold", color="#16324f")
    parametros = [
        (accuracy, "Accuracy (%)", "Accuracy: melhor em k = 9", "#176B9C", (94, 98.4), [94, 95, 96, 97, 98]),
        (f1, "F1 de M", "F1 de M: melhor em k = 9", "#18856b", (0.920, 0.975), [0.92, 0.94, 0.96]),
        (recall, "Recall de M", "Recall de M: melhor em k = 7", "#9b4dca", (0.88, 0.955), [0.88, 0.90, 0.92, 0.94]),
    ]
    for ax, (valores, ylabel, title, cor, limites, ticks) in zip(axes, parametros):
        ax.plot(ks, valores, "o-", color=cor, lw=2.3, ms=6, zorder=3)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10, color="#16324f")
        ax.set_ylim(*limites)
        ax.set_xlim(0, 32)
        ax.set_yticks(ticks, [f"{v:g}".replace(".", ",") for v in ticks])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e2e8f0")
        ax.grid(axis="x", color="#f1f5f9")
        ax.set_axisbelow(True)
        i = valores.index(max(valores))
        ax.scatter([ks[i]], [valores[i]], marker="*", s=220, color="#d45c28",
                   edgecolor="white", linewidth=0.8, zorder=5)
        texto = f"k = {ks[i]} → " + (f"{valores[i]:.4f}%" if ylabel == "Accuracy (%)" else f"{valores[i]:.3f}")
        ax.annotate(texto.replace(".", ","), (ks[i], valores[i]), xytext=(58, 15),
                    textcoords="offset points", color="#9a3d18", fontsize=11, weight="bold",
                    arrowprops={"arrowstyle": "->", "color": "#d45c28"})
    # Rótulos de Accuracy tornam as oscilações verificáveis sem consultar a tabela.
    for k, a in zip(ks, accuracy):
        if k == 9:
            continue
        axes[0].annotate(f"{a:.2f}".replace(".", ","), (k, a),
                         xytext=(0, -17 if k in (5, 11, 15, 25) else 9),
                         textcoords="offset points", ha="center", fontsize=9, color="#176B9C")
    axes[-1].set_xticks(ks, [str(k) for k in ks])
    axes[-1].set_xlabel("k — número de vizinhos", fontsize=12, labelpad=10)
    fig.text(0.10, 0.102, "Pontos = configurações avaliadas; linhas = ligação visual entre os pontos. Escalas verticais ampliadas.",
             fontsize=10, color="#475569")
    fig.text(0.10, 0.065, "Os dados mostram a variação nesta validação; pequenas diferenças não demonstram superioridade estatística.",
             fontsize=10, color="#475569")
    fig.text(0.10, 0.028, "Fonte: resultados.json, métricas recalculadas das matrizes de confusão. O teste reservado de 10% não foi utilizado.",
             fontsize=9, color="#475569")
    for extensao in ("png", "svg"):
        fig.savefig(pasta / f"desempenho_KNN_por_k.{extensao}", dpi=180)
    print("Gráfico gerado: 10 configurações; melhores confirmados: Accuracy/F1 k=9, Recall k=7.")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
