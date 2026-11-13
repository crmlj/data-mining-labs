# Data Mining com Weka

Experiências de classificação dos Labs 7 e 8. Em ambos os laboratórios, a seleção de modelos usou validação cruzada estratificada de 10 folds no conjunto de treino. O teste reservado foi usado para avaliar a configuração escolhida.

| Laboratório | Modelo selecionado | Treino / teste | Accuracy em validação cruzada | Accuracy no teste |
| --- | --- | ---: | ---: | ---: |
| [Lab 7 — WDBC](Lab7/Relatorio_Lab7/Relatorio_Tecnico_Lab7.md) | SMO com normalização, C=1 e PolyKernel de expoente 2 | 512 / 57 | 97,85% | 98,25% (56/57) |
| [Lab 8 — Ionosphere](Lab8/relatorio/Relatorio_Lab8_Ensemble.md) | Vote por maioria com seis classificadores, sem J48 | 281 / 70 | 95,73% | 91,43% (64/70) |

## Lab 7 — WDBC

Foram comparados ZeroR, OneR, dez valores de k para o IBk e 30 configurações de SMO. A seleção deu prioridade ao F1 da classe maligna (`M`), com accuracy como desempate. No teste, o SMO detetou 20 dos 21 casos malignos e classificou corretamente todos os 36 casos benignos.

- [Resultados do k-NN](Lab7/KNN/LEIA-ME.md) e [gráfico de desempenho por k](Lab7/KNN/desempenho_KNN_por_k.svg)
- [Grelha de parâmetros do SMO](Lab7/SMO/LEIA-ME.md) e [gráfico do efeito na accuracy](Lab7/SMO/Graficos_Parametros/01_efeitos_parametros_accuracy.svg)
- [Seleção e avaliação no teste](Lab7/Avaliacao_Final/LEIA-ME.md)

## Lab 8 — Ionosphere

O relatório inclui os resultados de ZeroR, OneR, J48, NaiveBayes, IBk, SMO, Bagging, RandomForest, AdaBoostM1, Vote e Stacking. A seleção deu prioridade à accuracy, com recall de `bad` como desempate. O Vote escolhido usa NaiveBayes, IBk com seleção automática até k=31, SMO com normalização e PolyKernel de expoente 2, RandomForest com 50 árvores, Bagging com 5000 REPTree e AdaBoostM1 com 19 DecisionStump.

No teste, o Vote identificou 19 de 25 casos `bad` e 45 de 45 casos `good`. O AUC ROC ponderado foi 0,880, com F1 de 0,864 para `bad` e de 0,938 para `good`. Há uma instância `bad` idêntica no treino e no teste, o que limita a independência da avaliação.

- [Vote com seis classificadores, validação cruzada](Lab8/vote/Vote_6_Majority.txt)
- [Vote com sete classificadores, validação cruzada](Lab8/vote/Vote_7_Majority.txt)
- [Vote com seis classificadores, avaliação final](Lab8/vote/Vote_6_teste_final.txt)

## Reproduzir as experiências

Usar o Weka 3.8.7 e colocar as partições ARFF locais nos [caminhos indicados em DADOS.md](DADOS.md). No Weka Explorer, abrir o ficheiro de treino em **Preprocess**, definir a classe (`Diagnosis`, primeira coluna no Lab 7, ou `class`, última coluna no Lab 8) e configurar o algoritmo em **Classify**. Cada saída do Weka regista a configuração em `Scheme` ou no comando. Para repetir a comparação, escolher **Cross-validation**, com 10 folds e seed 1. Para a avaliação final, escolher **Supplied test set** e carregar a partição de teste correspondente.

Os scripts abaixo usam Python 3 para iniciar o Weka. Por omissão, usam os caminhos do pacote Weka 3.8.7 para macOS. Noutras instalações, definir `WEKA_JAVA` para o executável Java compatível com o Weka e `WEKA_JAR` para o ficheiro `weka.jar`. Os scripts do Lab 7 regravam as respetivas saídas, pelo que convém executá-los numa cópia se for necessário preservar os ficheiros do laboratório.

```sh
python3 Lab7/KNN/executar_ibk_weka.py --train Lab7/wdbc-train.arff
python3 Lab7/Avaliacao_Final/executar_avaliacao_final.py --train Lab7/wdbc-train.arff --test Lab7/wdbc-test.arff
python3 Lab8/adaboost/executar_adaboost.py
```

O script AdaBoost valida e preserva as saídas já presentes. Para repetir o treino, executar o script numa cópia e retirar primeiro dessa cópia os ficheiros `AdaBoostM1_I*.txt`. Os comandos arquivados foram ajustados para omitir os caminhos pessoais da máquina original.
