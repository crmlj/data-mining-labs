# Ensemble – Comparação de classificadores no Ionosphere

O Vote com seis classificadores e votação por maioria apresentou a maior accuracy na validação cruzada de 10 folds, com 95,73%. No teste final, acertou 64 dos 70 exemplos reservados (19 bad e 45 good), obtendo 91,43%.

O conjunto original tem 351 instâncias e 34 atributos de entrada. A classe bad identifica retornos sem evidência de estrutura na ionosfera e good identifica retornos com essa evidência. A divisão estratificada foi feita no Weka usando StratifiedRemoveFolds, com numFolds=5, fold=2 e seed=1. Foram assim reservadas 281 instâncias para treino, das quais 101 bad e 180 good, e 70 para teste, das quais 25 bad e 45 good.

No IBk, a seleção automática avaliou K entre 1 e 31 (intervalo baseado nos ensaios do Lab 7). No Stacking, os seis modelos base (NaiveBayes, IBk, SMO, RandomForest, Bagging e AdaBoostM1) geraram as previsões para treinar o segundo nível (NaiveBayes, J48 ou BayesNet). Essas previsões foram produzidas em 10 folds internos, treinando os modelos em nove folds e fazendo previsões para o restante.

Os métodos testados (ZeroR, OneR, J48, NaiveBayes, IBk, SMO, Bagging, RandomForest, AdaBoostM1, Vote e Stacking) foram comparados por validação cruzada de 10 folds nas 281 instâncias de treino. A seleção privilegiou a maior accuracy e usou o recall de bad para desempatar. O conjunto de teste foi reservado para a avaliação final, depois da escolha do Vote.

No Bagging, 1000 e 5000 árvores atingiram 93,24% de accuracy, com uma pequena diferença no recall de bad. No RandomForest, 50 árvores deram o melhor resultado entre os três valores testados, com 95,37%. No AdaBoostM1, a maior accuracy ocorreu com 19 iterações, com 93,95%.

Retirar o J48 do Vote de sete classificadores aumentou a accuracy de 94,66% para 95,73%. Entre os dois metaclassificadores pedidos para o Stacking, NaiveBayes superou J48 em accuracy e recall de bad. O Stacking com NaiveBayes reconheceu 92 dos 101 exemplos bad avaliados por validação cruzada, com recall de 0,911.

No teste final, o modelo reconheceu 19 dos 25 exemplos bad e todos os 45 exemplos good. Os seis erros foram exemplos bad classificados como good. A accuracy foi de 91,43%, comparada com 95,73% na validação cruzada. O recall de bad foi de 76,00% no teste e de 89,11% na validação cruzada.

Foi identificado um registo bad repetido entre treino e teste, com os mesmos 34 atributos e classe. Essa duplicação limita a independência da avaliação final.

## Resultados no treino

Validação cruzada estratificada de 10 folds nas 281 instâncias de treino. AUC corresponde à ROC Area ponderada por classe apresentada pelo Weka. F1 bad e F1 good correspondem às classes b e g do ficheiro ARFF.

| Algoritmo | Informação e parâmetros | Accuracy | AUC | F1 bad | F1 good |
| --- | --- | --- | --- | --- | --- |
| ZeroR | Padrão | 64,06% | 0,496 | n.d. | 0,781 |
| OneR | B=6, padrão | 80,43% | 0,782 | 0,721 | 0,849 |
| J48 | C=0,25, M=2, padrão | 87,90% | 0,869 | 0,817 | 0,910 |
| NaiveBayes | Padrão | 84,34% | 0,938 | 0,804 | 0,870 |
| IBk | K=1, padrão | 86,48% | 0,823 | 0,782 | 0,902 |
| IBk | K=9 | 81,85% | 0,916 | 0,683 | 0,873 |
| IBk | Kmax=31, seleção automática interna (crossValidate = True) | 89,32% | 0,873 | 0,839 | 0,920 |
| SMO | C=1, Poly E=1, normalização, padrão | 87,19% | 0,830 | 0,793 | 0,907 |
| SMO | C=1, Poly E=2, normalização | 90,04% | 0,870 | 0,846 | 0,926 |
| Bagging | REPTree, I=10, amostra=100% | 91,46% | 0,960 | 0,875 | 0,935 |
| Bagging | REPTree, I=50, amostra=100% | 92,17% | 0,960 | 0,887 | 0,940 |
| Bagging | REPTree, I=100, amostra=100% | 92,88% | 0,956 | 0,897 | 0,946 |
| Bagging | REPTree, I=500, amostra=100% | 92,88% | 0,955 | 0,897 | 0,946 |
| Bagging | REPTree, I=1000, amostra=100% | 93,24% | 0,953 | 0,902 | 0,949 |
| Bagging | REPTree, I=5000, amostra=100% | 93,24% | 0,950 | 0,903 | 0,948 |
| RandomForest | I=10, restantes parâmetros padrão | 93,95% | 0,968 | 0,914 | 0,953 |
| RandomForest | I=50, restantes parâmetros padrão | 95,37% | 0,969 | 0,933 | 0,965 |
| RandomForest | I=100, restantes parâmetros padrão | 94,31% | 0,970 | 0,918 | 0,956 |
| Stacking | S6, meta NaiveBayes, folds internos=10 | 93,95% | 0,951 | 0,915 | 0,953 |
| Stacking | S6, meta J48, folds internos=10 | 91,46% | 0,896 | 0,876 | 0,935 |
| Stacking | S6, meta BayesNet, folds internos=10 | 94,31% | 0,946 | 0,919 | 0,956 |
| Vote | S7, votação por maioria | 94,66% | 0,930 | 0,921 | 0,960 |
| Vote | S6, votação por maioria | 95,73% | 0,943 | 0,938 | 0,968 |

I indica numIterations. S6 e S7 são definidos na secção sobre as combinações do Vote. n.d. assinala um F1 indefinido no Weka, porque o ZeroR prevê apenas good.

## AdaBoost com diferentes iterações

AdaBoostM1 com DecisionStump no primeiro nível, seed=1 e restantes parâmetros padrão. Todas as linhas correspondem à validação cruzada de 10 folds no treino.

| Algoritmo | Informação e parâmetros | Accuracy | AUC | F1 bad | F1 good |
| --- | --- | --- | --- | --- | --- |
| AdaBoostM1 | DecisionStump, I=1 | 82,56% | 0,726 | 0,684 | 0,880 |
| AdaBoostM1 | DecisionStump, I=2 | 82,56% | 0,865 | 0,684 | 0,880 |
| AdaBoostM1 | DecisionStump, I=3 | 90,39% | 0,865 | 0,864 | 0,926 |
| AdaBoostM1 | DecisionStump, I=4 | 85,41% | 0,882 | 0,748 | 0,897 |
| AdaBoostM1 | DecisionStump, I=5 | 90,39% | 0,882 | 0,864 | 0,926 |
| AdaBoostM1 | DecisionStump, I=6 | 88,97% | 0,912 | 0,821 | 0,920 |
| AdaBoostM1 | DecisionStump, I=7 | 89,32% | 0,921 | 0,837 | 0,921 |
| AdaBoostM1 | DecisionStump, I=8 | 90,39% | 0,908 | 0,851 | 0,929 |
| AdaBoostM1 | DecisionStump, I=9 | 89,68% | 0,928 | 0,838 | 0,924 |
| AdaBoostM1 | DecisionStump, I=10 | 89,32% | 0,929 | 0,833 | 0,921 |
| AdaBoostM1 | DecisionStump, I=11 | 92,17% | 0,939 | 0,882 | 0,941 |
| AdaBoostM1 | DecisionStump, I=12 | 91,81% | 0,941 | 0,876 | 0,939 |
| AdaBoostM1 | DecisionStump, I=13 | 92,17% | 0,949 | 0,880 | 0,942 |
| AdaBoostM1 | DecisionStump, I=14 | 91,46% | 0,945 | 0,870 | 0,937 |
| AdaBoostM1 | DecisionStump, I=15 | 92,53% | 0,958 | 0,886 | 0,944 |
| AdaBoostM1 | DecisionStump, I=16 | 92,53% | 0,946 | 0,886 | 0,944 |
| AdaBoostM1 | DecisionStump, I=17 | 93,24% | 0,946 | 0,898 | 0,949 |
| AdaBoostM1 | DecisionStump, I=18 | 92,53% | 0,952 | 0,888 | 0,944 |
| AdaBoostM1 | DecisionStump, I=19 | 93,95% | 0,945 | 0,910 | 0,954 |
| AdaBoostM1 | DecisionStump, I=20 | 92,88% | 0,954 | 0,894 | 0,947 |

A maior accuracy do AdaBoost ocorreu com 19 iterações, com 264 acertos em 281. A maior AUC desta série ocorreu com 15 iterações, com 0,958. O desempenho variou ao longo das iterações.

## Combinações do Vote e teste final

S7 reúne J48 padrão, NaiveBayes padrão, IBk automático com Kmax=31, SMO normalizado com C=1 e Poly E=2, RandomForest com 50 árvores, Bagging com 5000 REPTree e AdaBoost com 19 DecisionStump. S6 contém os mesmos modelos após retirar J48.

| Algoritmo | Informação e parâmetros | Accuracy | AUC | F1 bad | F1 good |
| --- | --- | --- | --- | --- | --- |
| Vote | S7, com J48, votação por maioria | 94,66% | 0,930 | 0,921 | 0,960 |
| Vote | S6, sem J48, votação por maioria | 95,73% | 0,943 | 0,938 | 0,968 |

Foram comparadas duas composições do Vote por votação por maioria. O conjunto S7 inclui o J48 e o conjunto S6 resulta da sua remoção. O S6 obteve a maior accuracy e foi selecionado para a avaliação no teste reservado.

## Avaliação final da configuração escolhida

| Algoritmo | Informação e parâmetros | Accuracy | AUC | F1 bad | F1 good |
| --- | --- | --- | --- | --- | --- |
| Vote | S6, votação por maioria, validação cruzada no treino | 95,73% | 0,943 | 0,938 | 0,968 |
| Vote | S6, votação por maioria, treino=281 e teste reservado=70 | 91,43% | 0,880 | 0,864 | 0,938 |

O teste contém 25 exemplos bad e 45 good. A matriz de confusão é [[19, 6], [0, 45]], com linhas reais e colunas previstas na ordem bad, good.
