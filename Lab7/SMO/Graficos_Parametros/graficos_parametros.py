"""Gráficos dos 30 resultados SMO do Weka — Lab 7.

Abre este ficheiro no editor Python ou no VS Code.
Instalação: python3 -m pip install matplotlib numpy
Execução:   python3 graficos_parametros.py
Janelas:    python3 graficos_parametros.py --show

Lê resultados_weka.json na mesma pasta. Não treina modelos, não utiliza
o conjunto de teste e não inclui a experiência SVC/libsvm do Python.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Abrir os gráficos em janelas")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    pasta = Path(__file__).resolve().parent
    dados = json.loads((pasta / "resultados_weka.json").read_text(encoding="utf-8"))
    resultados = dados["evaluations"]
    # Cada ponto corresponde a uma configuração efetivamente avaliada no Weka.
    indice = {(r["normalization"], r["kernel"], float(r["parameter"]), float(r["C"])): r
              for r in resultados}
    esperados = {(n, k, p, c) for n in ("semNorm", "norm")
                 for k, ps in (("Poly", (1, 2)), ("RBF", (0.001, 1, 100)))
                 for p in ps for c in (0.01, 1, 100)}
    assert len(resultados) == 30 and set(indice) == esperados, "Grelha de resultados incompleta"

    def resultado(norm, kernel, parametro, c):
        return indice[(norm, kernel, float(parametro), float(c))]

    def accuracy(norm, kernel, parametro, c):
        r = resultado(norm, kernel, parametro, c)
        m = r["confusion_matrix_M_B"]
        total = sum(m["M"]) + sum(m["B"])
        valor = 100 * (m["M"][0] + m["B"][1]) / total
        assert total == 512 and abs(valor - r["accuracy_percent"]) < 0.0001
        return valor

    def recall(norm, kernel, parametro, c):
        # Recall = malignos corretamente identificados / todos os malignos.
        m = resultado(norm, kernel, parametro, c)["confusion_matrix_M_B"]["M"]
        assert sum(m) == 191
        return 100 * m[0] / sum(m)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
                         "savefig.facecolor": "#f8fafc"})
    azul, laranja = "#176B9C", "#D45C28"
    cs = [0.01, 1, 100]
    gammas = [0.001, 1, 100]
    base = 100 * 321 / 512  # ZeroR: prever sempre a classe maioritária B.

    # FIGURA 1: todos os 30 resultados de Accuracy.
    # Eixos logarítmicos: as distâncias representam multiplicações.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.15,
                        top=0.84, hspace=0.52, wspace=0.21)
    fig.suptitle("Como os parâmetros alteraram os resultados do SVM", x=0.075,
                 y=0.975, ha="left", fontsize=21, weight="bold", color="#16324f")
    fig.text(0.075, 0.925, "30 configurações reais do Weka • 512 exemplos • validação cruzada de 10 folds",
             color="#475569", fontsize=12)
    fig.text(0.075, 0.885, "Accuracy = percentagem total de acertos. Quanto maior, melhor nesta métrica.",
             fontsize=11)

    def preparar(ax, xs, labels, xlabel):
        ax.set_xscale("log")
        ax.set_xticks(xs, labels)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(60, 101)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#e2e8f0")
        ax.axhline(base, color="#64748b", ls=":", lw=1.4, zorder=1)
        ax.set_axisbelow(True)

    for e, ax in zip((1, 2), axes[0]):
        preparar(ax, cs, ["0,01", "1", "100"], "C — penalização dos erros")
        ax.set_title(f"PolyKernel: expoente {e}" + (" (linear)" if e == 1 else " (quadrático)"), pad=15)
        for norm, label, cor, marker in (("semNorm", "Sem normalização", laranja, "s"),
                                         ("norm", "Com normalização", azul, "o")):
            ys = [accuracy(norm, "Poly", e, c) for c in cs]
            ax.plot(cs, ys, color=cor, marker=marker, ms=7, lw=2.2, label=label)
            for x, y in zip(cs, ys):
                ax.annotate(f"{y:.2f}".replace(".", ","), (x, y),
                            xytext=(0, 9 if norm == "norm" else -17),
                            textcoords="offset points", ha="center", fontsize=9, color=cor)
        ax.legend(loc="lower right", fontsize=10, frameon=True)

    cores = ["#8b5cf6", "#176B9C", "#18856b"]
    # Cores/traços são os mesmos nos dois painéis RBF.
    for norm, ax, titulo in (("semNorm", axes[1, 0], "RBF: sem normalização"),
                             ("norm", axes[1, 1], "RBF: com normalização")):
        preparar(ax, gammas, ["0,001", "1", "100"], "Gamma — alcance da influência de cada exemplo")
        ax.set_title(titulo, pad=15)
        for c, cor, marker, ls, size in zip(cs, cores, ("o", "s", "D"),
                                          ("-", "--", ":"), (10, 7, 4)):
            ax.plot(gammas, [accuracy(norm, "RBF", g, c) for g in gammas],
                    color=cor, marker=marker, ls=ls, ms=size, lw=2.2,
                    label=f"C = {c:g}".replace(".", ","))
        ax.legend(loc="upper right", fontsize=10)

    axes[0, 1].annotate("Melhor: 97,85%", (1, accuracy("norm", "Poly", 2, 1)),
                         xytext=(-65, -90), textcoords="offset points", fontsize=10,
                         color=azul, arrowprops={"arrowstyle": "->", "color": azul},
                         bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "pad": 4})
    fig.text(0.075, 0.067, "Linha cinzenta pontilhada: ZeroR = 62,70%. Algumas curvas coincidem exatamente.",
             fontsize=10, color="#475569")
    fig.text(0.075, 0.035, "As linhas ligam apenas valores testados. Estes resultados não demonstram que aumentar um parâmetro melhora sempre o modelo.",
             fontsize=10, color="#475569")
    for ext in ("png", "svg"):
        fig.savefig(pasta / f"01_efeitos_parametros_accuracy.{ext}", dpi=180)

    # FIGURA 2: todos os 30 recalls, calculados diretamente das matrizes.
    # Mesma escala 0–100% nos dois painéis para permitir a comparação.
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6.8))
    fig2.subplots_adjust(left=0.16, right=0.90, top=0.76, bottom=0.22, wspace=0.65)
    fig2.suptitle("Quantos malignos o modelo conseguiu identificar?", x=0.055,
                  y=0.96, ha="left", fontsize=21, weight="bold", color="#16324f")
    fig2.text(0.055, 0.885, "Recall de M (%) = malignos identificados / 191 malignos. Quanto maior, menos casos M ficam por identificar.",
              fontsize=11, color="#475569")
    linhas = [("Poly", 1, "Poly · expoente 1"), ("Poly", 2, "Poly · expoente 2"),
              ("RBF", 0.001, "RBF · gamma 0,001"), ("RBF", 1, "RBF · gamma 1"),
              ("RBF", 100, "RBF · gamma 100")]
    for norm, ax, titulo in (("semNorm", axes2[0], "Sem normalização"),
                             ("norm", axes2[1], "Com normalização")):
        valores = np.array([[recall(norm, k, p, c) for c in cs] for k, p, _ in linhas])
        im = ax.imshow(valores, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_title(titulo, pad=16)
        ax.set_xticks(range(3), ["0,01", "1", "100"])
        ax.set_yticks(range(5), [label for _, _, label in linhas])
        ax.set_xlabel("C")
        ax.tick_params(length=0, pad=9)
        for i in range(5):
            for j in range(3):
                ax.text(j, i, f"{valores[i, j]:.1f}%".replace(".", ","),
                        ha="center", va="center", weight="bold", fontsize=12,
                        color="white" if valores[i, j] > 65 else "#16324f")
    cax = fig2.add_axes([0.93, 0.22, 0.015, 0.54])
    fig2.colorbar(im, cax=cax, ticks=[0, 25, 50, 75, 100], label="Recall M (%)")
    fig2.text(0.055, 0.10, "0,0% = nenhum maligno identificado. Uma Accuracy de 62,70% pode coexistir com Recall de M igual a zero.", fontsize=11)
    fig2.text(0.055, 0.05, "Fonte: apenas os relatórios Weka do treino. O conjunto de teste de 10% e o resultado SVC/Python estão excluídos.",
              fontsize=10, color="#475569")
    for ext in ("png", "svg"):
        fig2.savefig(pasta / f"02_recall_malignos.{ext}", dpi=180)
    print("30 configurações verificadas. Gráficos guardados em:", pasta)
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
