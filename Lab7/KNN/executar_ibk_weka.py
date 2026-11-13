"""Executa o IBk do Weka instalado. Python apenas inicia Java e verifica as saídas.

Uso: python3 executar_ibk_weka.py --train /caminho/wdbc-train.arff
Saídas guardadas na mesma pasta deste script. Não utiliza o conjunto de teste.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import os
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

KS = [1, 3, 5, 7, 9, 11, 15, 19, 25, 31]
EXPECTED_SHA = "9a2d2930ecb72eaf8320c33b1846f23b02d3aaeed2d95a1fd4da8ad1170d290f"
JAVA = os.environ.get("WEKA_JAVA", "/Applications/weka-3.8.7.app/Contents/runtime/Contents/Home/bin/java")
JAR = os.environ.get("WEKA_JAR", "/Applications/weka-3.8.7.app/Contents/app/weka.jar")
SEARCH = 'weka.core.neighboursearch.LinearNNSearch -A "weka.core.EuclideanDistance -R first-last"'


def num(s):
    return float(s.replace(",", "."))


def verify(stdout, k):
    assert stdout.count("=== Stratified cross-validation ===") == 1
    assert re.search(rf"^Options: -K {k} -W 0 ", stdout, re.M)
    cv = stdout.split("=== Stratified cross-validation ===")[1]
    correct = re.search(r"Correctly Classified Instances\s+(\d+)\s+([\d.,]+)\s*%", cv)
    wrong = re.search(r"Incorrectly Classified Instances\s+(\d+)\s+([\d.,]+)\s*%", cv)
    total = re.search(r"Total Number of Instances\s+(\d+)", cv)
    assert correct and wrong and total and int(total[1]) == 512
    assert int(correct[1]) + int(wrong[1]) == 512
    assert abs(num(correct[2]) - 100 * int(correct[1]) / 512) <= 0.000051
    assert abs(num(wrong[2]) - 100 * int(wrong[1]) / 512) <= 0.000051
    matrix_text = cv.split("=== Confusion Matrix ===")[1]
    matches = re.findall(r"^\s*(\d+)\s+(\d+)\s*\|\s*(\w+)\s*=\s*([MB])\s*$", matrix_text, re.M)
    assert len(matches) == 2
    assert {code: label for a, b, code, label in matches} == {"a": "M", "b": "B"}
    matrix = {label: [int(a), int(b)] for a, b, code, label in matches}
    assert sum(matrix["M"]) == 191 and sum(matrix["B"]) == 321
    assert matrix["M"][0] + matrix["B"][1] == int(correct[1])
    detail = cv.split("=== Detailed Accuracy By Class ===")[1].split("=== Confusion Matrix ===")[0]
    lines = {s.split()[-1]: s.split() for s in detail.splitlines()
             if s.split() and s.split()[-1] in {"M", "B"}}
    assert set(lines) == {"M", "B"}
    exact = {}
    for label, pos in (("M", 0), ("B", 1)):
        neg = "B" if label == "M" else "M"
        tp, fn = matrix[label][pos], matrix[label][1 - pos]
        fp, tn = matrix[neg][pos], matrix[neg][1 - pos]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp else None
        for raw, value in zip(lines[label][:5], [recall, fp / (fp + tn), precision, recall, f1]):
            assert (raw == "?" and value is None) or (raw != "?" and value is not None and abs(num(raw) - value) < 0.000501)
        exact[label] = dict(precision=precision, recall=recall, F1=f1)
    return dict(k=k, accuracy_percent=num(correct[2]), correct=int(correct[1]),
                incorrect=int(wrong[1]), precision_M=num(lines["M"][2]),
                recall_M=num(lines["M"][3]), F1_M=num(lines["M"][4]),
                exact_metrics_M=exact["M"], confusion_matrix_M_B=matrix, verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    args = parser.parse_args()
    train = args.train.resolve()
    raw = train.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == EXPECTED_SHA, "O ficheiro não corresponde ao treino auditado do Lab 7"
    text = raw.decode("utf-8")
    assert len(re.findall(r"^@attribute\s", text, re.M | re.I)) == 31
    data = re.split(r"(?im)^@data\s*$", text)[1]
    counts = Counter(s.split(",")[0] for s in data.splitlines() if s.strip() and not s.lstrip().startswith("%"))
    assert counts == {"M": 191, "B": 321}
    out = Path(__file__).resolve().parent
    rows = []
    for k in KS:
        command = [JAVA, "--add-opens", "java.base/java.lang=ALL-UNNAMED", "-Djava.awt.headless=true",
                   "-Xmx1024m", "-cp", JAR, "weka.classifiers.lazy.IBk", "-t", str(train),
                   "-c", "1", "-x", "10", "-s", "1", "-v", "-K", str(k), "-W", "0", "-A", SEARCH]
        start = time.monotonic()
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        row = verify(run.stdout, k)
        name = f"IBk_K{k}.txt"
        header = ("=== Run information ===\n"
                  "Scheme: weka.classifiers.lazy.IBk\n"
                  "Dataset: wdbc-train.arff; 512 instances; 191 M / 321 B\n"
                  "Attributes: 31; Diagnosis is class (attribute 1); 30 numeric predictors; ID removed\n"
                  "Positive class: M\n"
                  "Test mode: stratified 10-fold cross-validation; evaluation seed 1\n"
                  "Distance: EuclideanDistance, internal normalization enabled (no -D)\n"
                  "Search: LinearNNSearch; distance weighting: none; windowSize: 0\n"
                  "Internal automatic k selection: disabled (no -X); requested k is fixed\n"
                  "Weka: 3.8.7 installed Java runtime; no Python classification model\n"
                  f"Dataset SHA256: {sha}\nCommand: {shlex.join(command)}\n\n")
        report = header + run.stdout
        (out / name).write_text(report, encoding="utf-8")
        row.update(source_file=name, source_file_sha256=hashlib.sha256(report.encode()).hexdigest(),
                   command=command, elapsed_seconds=round(time.monotonic() - start, 3),
                   completed_at=datetime.now().astimezone().isoformat())
        if run.stderr.strip():
            (out / f"IBk_K{k}.stderr.log").write_text(run.stderr, encoding="utf-8")
        rows.append(row)
        print(f'k={k}: Accuracy {row["accuracy_percent"]:.4f}%; P {row["precision_M"]:.3f}; R {row["recall_M"]:.3f}; F1 {row["F1_M"]:.3f}', flush=True)
    assert [r["k"] for r in rows] == KS
    best_acc = max(r["correct"] for r in rows)
    best_f1 = max(r["exact_metrics_M"]["F1"] for r in rows)
    best_recall = max(r["exact_metrics_M"]["recall"] for r in rows)
    best = dict(accuracy_k=[r["k"] for r in rows if r["correct"] == best_acc],
                F1_M_k=[r["k"] for r in rows if r["exact_metrics_M"]["F1"] == best_f1],
                recall_M_k=[r["k"] for r in rows if r["exact_metrics_M"]["recall"] == best_recall])
    summary = dict(weka_version="3.8.7", classifier="weka.classifiers.lazy.IBk", dataset="wdbc-train.arff",
                   dataset_sha256=sha, instances=512, class_counts=dict(counts), positive_class="M",
                   folds=10, seed=1, distance="EuclideanDistance", distance_normalization=True,
                   distance_weighting="none", internal_k_selection=False, search="LinearNNSearch",
                   completed=10, evaluations=rows, best=best, holdout_used=False)
    (out / "resultados.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    def dec(value, places):
        return f"{value:.{places}f}".replace(".", ",")
    lines = ["# Lab 7 — KNN / IBk: comparação de k", "",
             "**10 configurações completas e verificadas no Weka 3.8.7.**", "",
             "Treino: wdbc-train.arff, 512 instâncias (191 M e 321 B), 30 preditores, ID removido.",
             "Classe positiva: **M**. Validação cruzada estratificada de **10 folds**, seed de avaliação **1**.",
             "O conjunto de teste reservado de 10% não foi utilizado.", "",
             "Apenas k varia. Pesquisa LinearNNSearch, distância Euclidiana com normalização interna ativa,",
             "sem ponderação por distância, windowSize=0 e seleção automática de k desativada.",
             "A normalização da distância é o comportamento padrão do Weka: a opção -D, que a desativa, não é usada.",
             "Referência: https://weka.sourceforge.io/doc.dev/weka/core/EuclideanDistance.html", "",
             "| k | Accuracy (%) | Precision M | Recall M | F1 M | Acertos / 512 | FN | FP |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r["confusion_matrix_M_B"]
        lines.append(f'| {r["k"]} | {dec(r["accuracy_percent"],4)} | {dec(r["precision_M"],3)} | {dec(r["recall_M"],3)} | {dec(r["F1_M"],3)} | {r["correct"]} | {m["M"][1]} | {m["B"][0]} |')
    lines += ["", f'**Melhor Accuracy:** k = {", ".join(map(str,best["accuracy_k"]))}, com {dec(100*best_acc/512,4)}%.',
              f'**Melhor F1 de M:** k = {", ".join(map(str,best["F1_M_k"]))}, com {dec(best_f1,3)}.',
              f'**Maior Recall de M:** k = {", ".join(map(str,best["recall_M_k"]))}, com {dec(best_recall,3)}.', "",
              "FN = malignos classificados como benignos. FP = benignos classificados como malignos.",
              "As métricas da tabela são as impressas pelo Weka; os melhores são determinados com valores exatos calculados das matrizes.",
              "As matrizes têm classes reais nas linhas e previstas nas colunas, ordem M/B.",
              "Accuracy, totais e Precision/Recall/F1 das duas classes foram verificados contra todas as matrizes.", "",
              "Cada IBk_K*.txt contém os parâmetros, o comando e a saída completa do Weka, incluindo a matriz de confusão.",
              "Python apenas inicia o Weka/Java e organiza as saídas; o classificador executado é o IBk do Weka.",
              "O script executar_ibk_weka.py permite reproduzir a execução passando --train com o caminho do treino.", "",
              "A comparação é feita no treino para selecionar candidatos. Pequenas diferenças nesta validação não demonstram superioridade estatística.",
              f"SHA256 do treino: `{sha}`", ""]
    (out / "LEIA-ME.md").write_text("\n".join(lines), encoding="utf-8")
    print("COMPLETE: 10 relatórios Weka verificados", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
