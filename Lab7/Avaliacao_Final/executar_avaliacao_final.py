"""Treina o SMO selecionado por CV e avalia o teste independente no Weka.

Uso: python3 executar_avaliacao_final.py --train /caminho/wdbc-train.arff --test /caminho/wdbc-test.arff
Mantém selecao_por_validacao_cruzada.json nesta pasta. Python só orquestra Java/Weka.
Não procura parâmetros no teste. A normalização é aprendida pelo SMO no treino.
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
from types import SimpleNamespace

JAVA = os.environ.get("WEKA_JAVA", "/Applications/weka-3.8.7.app/Contents/runtime/Contents/Home/bin/java")
JAR = os.environ.get("WEKA_JAR", "/Applications/weka-3.8.7.app/Contents/app/weka.jar")
TRAIN_SHA = '9a2d2930ecb72eaf8320c33b1846f23b02d3aaeed2d95a1fd4da8ad1170d290f'


def number(s):
    return float(s.replace(',', '.'))


def read_dataset(path, counts):
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    header, data = re.split(r'(?im)^@data\s*$', text)
    attributes = re.findall(r'^@attribute\s+(.+)$', header, re.M | re.I)
    rows = [line.strip() for line in data.splitlines() if line.strip() and not line.lstrip().startswith('%')]
    assert len(attributes) == 31
    assert Counter(row.split(',')[0] for row in rows) == counts
    return dict(sha256=hashlib.sha256(raw).hexdigest(), rows=rows, attributes=attributes)


def verify_test(stdout):
    marker = re.search(r'=== (?:Error|Evaluation) on test data ===', stdout)
    assert marker
    assert '=== Stratified cross-validation ===' not in stdout
    cv = stdout[marker.end():]
    correct = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.,]+)\s*%', cv)
    wrong = re.search(r'Incorrectly Classified Instances\s+(\d+)\s+([\d.,]+)\s*%', cv)
    total = re.search(r'Total Number of Instances\s+(\d+)', cv)
    assert correct and wrong and total and int(total[1]) == 57
    assert int(correct[1]) + int(wrong[1]) == 57
    assert abs(number(correct[2]) - 100 * int(correct[1]) / 57) < 0.000051
    assert abs(number(wrong[2]) - 100 * int(wrong[1]) / 57) < 0.000051
    matches = re.findall(r'^\s*(\d+)\s+(\d+)\s*\|\s*(\w+)\s*=\s*([MB])\s*$', cv.split('=== Confusion Matrix ===')[1], re.M)
    assert len(matches) == 2
    assert {code: label for a, b, code, label in matches} == {'a': 'M', 'b': 'B'}
    matrix = {label: [int(a), int(b)] for a, b, code, label in matches}
    assert sum(matrix['M']) == 21 and sum(matrix['B']) == 36
    assert matrix['M'][0] + matrix['B'][1] == int(correct[1])
    detail = cv.split('=== Detailed Accuracy By Class ===')[1].split('=== Confusion Matrix ===')[0]
    lines = {s.split()[-1]: s.split() for s in detail.splitlines() if s.split() and s.split()[-1] in {'M', 'B'}}
    assert set(lines) == {'M', 'B'}
    exact = {}
    for label, pos in (('M', 0), ('B', 1)):
        neg = 'B' if label == 'M' else 'M'
        tp, fn = matrix[label][pos], matrix[label][1-pos]
        fp, tn = matrix[neg][pos], matrix[neg][1-pos]
        p, r = tp / (tp+fp), tp / (tp+fn)
        f = 2*tp / (2*tp+fp+fn)
        for raw, expected in zip(lines[label][:5], [r, fp/(fp+tn), p, r, f]):
            assert abs(number(raw)-expected) < 0.000501
        exact[label] = dict(precision=p, recall=r, F1=f)
    return dict(instances=57, correct=int(correct[1]), incorrect=int(wrong[1]),
                accuracy_percent=number(correct[2]), exact_accuracy_percent=100*int(correct[1])/57,
                precision_M=number(lines['M'][2]), recall_M=number(lines['M'][3]), F1_M=number(lines['M'][4]),
                exact_metrics_M=exact['M'], confusion_matrix_M_B=matrix, verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--test', required=True, type=Path)
    parser.add_argument('--reuse-existing', action='store_true', help='Verifica a saída e o modelo já guardados, sem novo treino')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    selection = json.loads((out/'selecao_por_validacao_cruzada.json').read_text())
    chosen = selection['selected']['result']
    assert chosen['name'] == 'SMO_norm_Poly_E2_C1'
    assert not selection['holdout_consulted_for_selection']
    train_path, test_path = args.train.resolve(), args.test.resolve()
    assert train_path != test_path
    train = read_dataset(train_path, {'M':191, 'B':321})
    test = read_dataset(test_path, {'M':21, 'B':36})
    assert train['sha256'] == TRAIN_SHA
    assert train['attributes'] == test['attributes']
    assert not (set(train['rows']) & set(test['rows'])), 'Instâncias partilhadas entre treino e teste'
    model = out/'SMO_norm_Poly_E2_C1.model'
    base = [JAVA, '--add-opens', 'java.base/java.lang=ALL-UNNAMED', '-Djava.awt.headless=true',
            '-Xmx1024m', '-cp', JAR, 'weka.classifiers.functions.SMO']
    # Com -T, o Weka já usa o teste fornecido; -no-cv é incompatível com -T.
    command = base + ['-t', str(train_path), '-T', str(test_path), '-c', '1', '-v', '-d', str(model),
                      '-C', '1', '-L', '0.001', '-P', '1.0E-12', '-N', '0', '-V', '-1', '-W', '1',
                      '-K', 'weka.classifiers.functions.supportVector.PolyKernel -E 2.0 -C 250007']
    report = out/'SMO_norm_Poly_E2_C1_teste_independente.txt'
    if args.reuse_existing:
        existing = report.read_text(encoding='utf-8')
        assert 'Command: '+shlex.join(command) in existing
        assert f'Train SHA256: {train["sha256"]}' in existing and f'Test SHA256: {test["sha256"]}' in existing
        run = SimpleNamespace(stdout=existing, stderr='', returncode=0)
    else:
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0 and 'Weka exception:' not in run.stderr, run.stdout + run.stderr
    header = ('=== Run information ===\nWeka: 3.8.7, installed application Java runtime\n'
              'Scheme: weka.classifiers.functions.SMO; C=1; Normalize (-N 0); PolyKernel exponent=2\n'
              'Selection: highest CV F1 of M; also highest CV accuracy; frozen before holdout evaluation\n'
              'Training: all 512 training instances; ID removed; Diagnosis is class attribute 1\n'
              'Independent test: 57 instances (21 M / 36 B); not used for model selection\n'
              'Preprocessing: SMO fits its internal normalization on training data and applies it to test data\n'
              f'Train SHA256: {train["sha256"]}\nTest SHA256: {test["sha256"]}\n'
              'Command: '+shlex.join(command)+'\n\n')
    if not args.reuse_existing:
        report.write_text(header+run.stdout, encoding='utf-8')
    if run.stderr.strip():
        (out/'weka.stderr.log').write_text(run.stderr, encoding='utf-8')
    elif (out/'weka.stderr.log').exists():
        (out/'weka.stderr.log').unlink()
    result = verify_test(run.stdout)
    assert model.exists() and model.stat().st_size > 0
    # Confirma que o ficheiro .model guardado pode ser carregado e reproduz a avaliação.
    reload_cmd = base + ['-l', str(model), '-T', str(test_path), '-c', '1', '-v']
    loaded = subprocess.run(reload_cmd, capture_output=True, text=True, timeout=120)
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    loaded_result = verify_test(loaded.stdout)
    assert loaded_result == result
    delta = dict(accuracy_percentage_points=result['exact_accuracy_percent']-100*chosen['correct']/512,
                 precision_M=result['exact_metrics_M']['precision']-chosen['exact_metrics_M']['precision'],
                 recall_M=result['exact_metrics_M']['recall']-chosen['exact_metrics_M']['recall'],
                 F1_M=result['exact_metrics_M']['F1']-chosen['exact_metrics_M']['F1'])
    summary = dict(weka_version='3.8.7', selection=selection, training_instances=512, test_instances=57,
                   train_sha256=train['sha256'], test_sha256=test['sha256'], train_test_rows_disjoint=True,
                   test_result=result, difference_test_minus_CV=delta, command=command,
                   model_file=model.name, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                   saved_model_reloaded_and_verified=True, source_report=report.name,
                   source_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                   completed_at=datetime.now().astimezone().isoformat())
    (out/'resultado_final.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    def dec(v, places=3):
        if isinstance(v,str): v=number(v)
        return f'{v:.{places}f}'.replace('.',',')
    lines = ['# Lab 7 — 7.3: seleção do modelo e avaliação independente', '',
             '## Comparação na validação cruzada', '',
             'Treino: 512 instâncias. Validação cruzada estratificada de 10 folds, seed 1. Classe positiva: M.',
             'Critério de seleção: maior F1 de M, usando Accuracy como desempate. O vencedor também apresenta a maior Accuracy.', '',
             '| Modelo candidato | Accuracy (%) | Precision M | Recall M | F1 M |',
             '|---|---:|---:|---:|---:|']
    for candidate in selection['candidates']:
        r=candidate['result']
        label='SMO: Normalize, PolyKernel E=2, C=1' if candidate['family']=='SMO' else 'KNN / IBk: k=9'
        lines.append(f'| {label} | {dec(r["accuracy_percent"],4)} | {dec(r["precision_M"])} | {dec(r["recall_M"])} | {dec(r["F1_M"])} |')
    lines += ['', '**Modelo selecionado: SMO, C=1, PolyKernel expoente 2, normalização ativa (-N 0).**',
              'A escolha foi registada antes da avaliação do teste independente. Não houve pesquisa de parâmetros no teste.',
              'O modelo foi treinado com todos os 512 exemplos do treino. A normalização interna do SMO foi ajustada no treino.', '',
              '## Validação cruzada versus teste independente', '',
              '| Avaliação | Instâncias | Accuracy (%) | Precision M | Recall M | F1 M |',
              '|---|---:|---:|---:|---:|---:|',
              f'| Validação cruzada no treino | 512 | {dec(chosen["accuracy_percent"],4)} | {dec(chosen["precision_M"])} | {dec(chosen["recall_M"])} | {dec(chosen["F1_M"])} |',
              f'| Teste independente | 57 | {dec(result["accuracy_percent"],4)} | {dec(result["precision_M"])} | {dec(result["recall_M"])} | {dec(result["F1_M"])} |', '',
              f'Diferença de Accuracy (teste menos CV): **{dec(delta["accuracy_percentage_points"],4)} pontos percentuais**.',
              f'Diferenças das métricas de M (teste menos CV): Precision {dec(delta["precision_M"])}, Recall {dec(delta["recall_M"])}, F1 {dec(delta["F1_M"])}.', '',
              '## Matriz de confusão do teste', '',
              'Classes reais nas linhas e previstas nas colunas:', '',
              '| Real / Previsto | M | B |', '|---|---:|---:|',
              f'| M | {result["confusion_matrix_M_B"]["M"][0]} | {result["confusion_matrix_M_B"]["M"][1]} |',
              f'| B | {result["confusion_matrix_M_B"]["B"][0]} | {result["confusion_matrix_M_B"]["B"][1]} |', '',
              f'Acertos: {result["correct"]}/57. Falsos negativos: {result["confusion_matrix_M_B"]["M"][1]}. Falsos positivos: {result["confusion_matrix_M_B"]["B"][0]}.', '',
              '## Interpretação e verificação', '',
              'A validação cruzada estima o desempenho durante a seleção; o teste avalia o modelo final em dados reservados.',
              'O teste pode apresentar resultados superiores ou inferiores aos da CV por variação da amostra.',
              'Com apenas 57 exemplos, cada erro altera a Accuracy em cerca de 1,75 pontos percentuais.',
              'Os resultados não permitem afirmar que um aumento no teste prova superioridade ou ausência de sobreajuste.',
              'Não se alteraram parâmetros após observar o teste. Accuracy e métricas de ambas as classes conferem com a matriz.',
              'Os conjuntos têm atributos compatíveis e não partilham linhas. O modelo guardado foi recarregado e reproduziu a avaliação.', '',
              '## Ficheiros', '',
              '- SMO_norm_Poly_E2_C1_teste_independente.txt: saída completa do Weka e comando.',
              '- SMO_norm_Poly_E2_C1.model: modelo treinado nos 512 exemplos, utilizável no Weka.',
              '- selecao_por_validacao_cruzada.json: decisão e candidatos registados antes do teste.',
              '- resultado_final.json: métricas, diferenças, hashes e verificações.',
              '- executar_avaliacao_final.py: reprodução com o motor Java do Weka.', '',
              f'Treino SHA256: `{train["sha256"]}`', f'Teste SHA256: `{test["sha256"]}`', '']
    (out/'LEIA-ME.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(test=result, difference_test_minus_CV=delta, saved_model_verified=True), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
