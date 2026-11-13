"""Executa AdaBoostM1 no Weka com 1 a 20 iterações e CV de 10 folds.

Uso: python3 executar_adaboost.py
Guarda cada resultado nesta pasta e preserva ficheiros que já existam.
"""

from pathlib import Path
from collections import Counter
import hashlib
import os
import re
import shlex
import subprocess


def main():
    folder = Path(__file__).resolve().parent
    training = folder.parent / "train_ionosphere.arff"
    java = Path(os.environ.get("WEKA_JAVA", "/Applications/weka-3.8.7.app/Contents/runtime/Contents/Home/bin/java"))
    jar = Path(os.environ.get("WEKA_JAR", "/Applications/weka-3.8.7.app/Contents/app/weka.jar"))
    for required in (training, java, jar):
        if not required.is_file():
            raise SystemExit(f"Ficheiro necessário em falta: {required}")

    training_bytes = training.read_bytes()
    training_sha = hashlib.sha256(training_bytes).hexdigest()
    data = re.split(r"(?im)^@data\s*$", training_bytes.decode("utf-8"), maxsplit=1)[1]
    rows = [row.strip() for row in data.splitlines() if row.strip() and not row.lstrip().startswith("%")]
    class_counts = Counter(row.rsplit(",", 1)[-1].strip() for row in rows)
    if class_counts != {"b": 101, "g": 180}:
        raise SystemExit(f"O treino difere do usado no Lab 8: {dict(class_counts)}")

    for iterations in range(1, 21):
        destination = folder / f"AdaBoostM1_I{iterations:02d}.txt"
        if destination.exists():
            previous = destination.read_text(encoding="utf-8")
            if (
                f"Training SHA256: {training_sha}\n" not in previous
                or f"numIterations: {iterations}; classifier seed: 1; base classifier: DecisionStump\n" not in previous
                or "=== Stratified cross-validation ===" not in previous
            ):
                raise SystemExit(f"Resultado existente precisa de verificação: {destination}")
            print(f"Preservado o resultado existente: {destination.name}", flush=True)
            continue

        command = [
            str(java), "--add-opens", "java.base/java.lang=ALL-UNNAMED",
            "-Djava.awt.headless=true", "-cp", str(jar),
            "weka.classifiers.meta.AdaBoostM1",
            "-t", str(training), "-c", "last", "-x", "10", "-s", "1", "-v",
            "-I", str(iterations), "-S", "1",
            "-W", "weka.classifiers.trees.DecisionStump",
        ]
        print(f"A executar {iterations}/20...", flush=True)
        run = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if run.returncode != 0 or "=== Stratified cross-validation ===" not in run.stdout:
            raise SystemExit(f"Falha com {iterations} iterações.\n{run.stderr}\n{run.stdout}")

        cv = run.stdout.split("=== Stratified cross-validation ===", 1)[1]
        accuracy = re.search(r"^Correctly Classified Instances\s+\d+\s+([\d.,]+)", cv, re.M)
        matrix_rows = re.findall(
            r"^\s*(\d+)\s+(\d+)\s*\|\s*\w+\s*=\s*([bg])\s*$", cv, re.M
        )
        matrix = {label: (int(first), int(second)) for first, second, label in matrix_rows}
        if accuracy is None or set(matrix) != {"b", "g"}:
            raise SystemExit(f"Não foi possível verificar o resultado de {iterations} iterações.")
        correct = int(re.search(r"^Correctly Classified Instances\s+(\d+)", cv, re.M)[1])
        total = int(re.search(r"^Total Number of Instances\s+(\d+)", cv, re.M)[1])
        if (
            total != len(rows)
            or sum(matrix["b"]) != class_counts["b"]
            or sum(matrix["g"]) != class_counts["g"]
            or matrix["b"][0] + matrix["g"][1] != correct
            or abs(float(accuracy[1].replace(",", ".")) - 100 * correct / total) > 0.000051
        ):
            raise SystemExit(f"Métricas inconsistentes com {iterations} iterações.")

        header = (
            "=== Run information ===\n"
            "Scheme: weka.classifiers.meta.AdaBoostM1\n"
            f"Training file: {training.name}\n"
            f"Training SHA256: {training_sha}\n"
            "Weka: 3.8.7\n"
            "Test mode: stratified 10-fold cross-validation; evaluation seed 1\n"
            f"numIterations: {iterations}; classifier seed: 1; base classifier: DecisionStump\n"
            f"Command: {shlex.join(command)}\n\n"
        )
        with destination.open("x", encoding="utf-8") as output:
            output.write(header + run.stdout)
        recall_b = matrix["b"][0] / sum(matrix["b"])
        print(
            f"Guardado {destination.name}: accuracy {accuracy[1]}%; recall de b {recall_b:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
