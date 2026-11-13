# Lab 7 — KNN / IBk: comparação de k

**10 configurações completas e verificadas no Weka 3.8.7.**

Treino: wdbc-train.arff, 512 instâncias (191 M e 321 B), 30 preditores, ID removido.
Classe positiva: **M**. Validação cruzada estratificada de **10 folds**, seed de avaliação **1**.
O conjunto de teste reservado de 10% não foi utilizado.

Apenas k varia. Pesquisa LinearNNSearch, distância Euclidiana com normalização interna ativa,
sem ponderação por distância, windowSize=0 e seleção automática de k desativada.
A normalização da distância é o comportamento padrão do Weka: a opção -D, que a desativa, não é usada.
Referência: https://weka.sourceforge.io/doc.dev/weka/core/EuclideanDistance.html

| k | Accuracy (%) | Precision M | Recall M | F1 M | Acertos / 512 | FN | FP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 94,5313 | 0,931 | 0,921 | 0,926 | 484 | 15 | 13 |
| 3 | 96,6797 | 0,978 | 0,932 | 0,954 | 495 | 13 | 4 |
| 5 | 96,8750 | 0,983 | 0,932 | 0,957 | 496 | 13 | 3 |
| 7 | 96,8750 | 0,978 | 0,937 | 0,957 | 496 | 12 | 4 |
| 9 | 97,0703 | 0,989 | 0,932 | 0,960 | 497 | 13 | 2 |
| 11 | 96,8750 | 0,994 | 0,921 | 0,957 | 496 | 15 | 1 |
| 15 | 96,4844 | 0,989 | 0,916 | 0,951 | 494 | 16 | 2 |
| 19 | 96,6797 | 0,994 | 0,916 | 0,954 | 495 | 16 | 1 |
| 25 | 95,7031 | 0,977 | 0,906 | 0,940 | 490 | 18 | 4 |
| 31 | 95,7031 | 0,988 | 0,895 | 0,940 | 490 | 20 | 2 |

**Melhor Accuracy:** k = 9, com 97,0703%.
**Melhor F1 de M:** k = 9, com 0,960.
**Maior Recall de M:** k = 7, com 0,937.

FN = malignos classificados como benignos. FP = benignos classificados como malignos.
As métricas da tabela são as impressas pelo Weka; os melhores são determinados com valores exatos calculados das matrizes.
As matrizes têm classes reais nas linhas e previstas nas colunas, ordem M/B.
Accuracy, totais e Precision/Recall/F1 das duas classes foram verificados contra todas as matrizes.

Cada IBk_K*.txt contém os parâmetros, o comando e a saída completa do Weka, incluindo a matriz de confusão.
Python apenas inicia o Weka/Java e organiza as saídas; o classificador executado é o IBk do Weka.
O script executar_ibk_weka.py permite reproduzir a execução passando --train com o caminho do treino.

A comparação é feita no treino para selecionar candidatos. Pequenas diferenças nesta validação não demonstram superioridade estatística.
SHA256 do treino: `9a2d2930ecb72eaf8320c33b1846f23b02d3aaeed2d95a1fd4da8ad1170d290f`
