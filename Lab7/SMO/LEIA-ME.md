# Lab 7 — verificação dos relatórios Weka e comparação SMO

**30 configurações SMO completas e verificadas**, mais os baselines ZeroR e OneR: **32 relatórios**.
A comparação utiliza apenas saídas do Weka. A experiência SVC/libsvm da pasta Python_SVM e o registo histórico de interrupção não entram na avaliação.

## Verificação realizada

- Todas as configurações SMO pedidas estão presentes: 15 sem normalização e 15 normalizadas.
- Os valores C, o kernel, o expoente/gamma e a normalização no conteúdo correspondem aos nomes dos 30 ficheiros.
- Todas as avaliações têm 512 instâncias, 191 M e 321 B, em validação cruzada estratificada de 10 folds.
- Accuracy, erros, Precision, Recall, F1 e taxas TP/FP das duas classes conferem com as matrizes de confusão, dentro do arredondamento impresso pelo Weka.
- As métricas registadas abaixo pertencem à classe positiva M. Não se usam as médias ponderadas nem a avaliação sobre os próprios dados de treino.
- Nos relatórios CLI foi confirmado o ficheiro de treino, ausência de conjunto de teste fornecido e SHA256 do dataset. O conjunto reservado de 10% continua fora desta comparação.
- Os 29 relatórios CLI documentam seed de avaliação 1. O relatório do Explorer e os baselines não imprimem essa seed; a sessão foi anteriormente configurada com seed 1. O parâmetro -W 1 do SMO é uma opção do classificador e não confirma, por si só, a seed da avaliação.

## Resultado que faltava

SMO sem normalização, PolyKernel expoente 1, C=100: **495/512 acertos (96,6797%)**. Precision M **0,958**, Recall M **0,953**, F1 M **0,955**.
Matriz (reais nas linhas, previstas nas colunas, ordem M/B): **[[182, 9], [8, 313]]**. São 182 malignos identificados, 9 falsos negativos e 8 falsos positivos.
O relatório atual substitui a condição de pendência. O ficheiro *_INTERROMPIDO.txt documenta apenas uma tentativa anterior e permanece como histórico.

## Baselines

| Modelo | Accuracy (%) | Precision M | Recall M | F1 M |
|---|---:|---:|---:|---:|
| zeroR | 62,6953 | ? | 0,000 | ? |
| oneR | 88,8672 | 0,881 | 0,812 | 0,845 |

## Todas as configurações SMO

| Normalização | Kernel | C | Expoente / gamma | Accuracy (%) | Precision M | Recall M | F1 M |
|---|---|---:|---:|---:|---:|---:|---:|
| Não | Poly | 1 | E=1 | 95,3125 | 0,956 | 0,916 | 0,936 |
| Não | Poly | 0.01 | E=1 | 94,9219 | 0,941 | 0,921 | 0,931 |
| Não | Poly | 100 | E=1 | 96,6797 | 0,958 | 0,953 | 0,955 |
| Não | Poly | 1 | E=2 | 92,1875 | 0,975 | 0,812 | 0,886 |
| Não | Poly | 0.01 | E=2 | 92,1875 | 0,975 | 0,812 | 0,886 |
| Não | Poly | 100 | E=2 | 92,1875 | 0,975 | 0,812 | 0,886 |
| Não | RBF | 1 | gamma=0.001 | 92,9688 | 0,897 | 0,916 | 0,907 |
| Não | RBF | 0.01 | gamma=0.001 | 62,6953 | ? | 0,000 | ? |
| Não | RBF | 100 | gamma=0.001 | 91,2109 | 0,869 | 0,901 | 0,884 |
| Não | RBF | 1 | gamma=1 | 62,6953 | ? | 0,000 | ? |
| Não | RBF | 0.01 | gamma=1 | 62,6953 | ? | 0,000 | ? |
| Não | RBF | 100 | gamma=1 | 62,6953 | ? | 0,000 | ? |
| Não | RBF | 1 | gamma=100 | 62,6953 | ? | 0,000 | ? |
| Não | RBF | 0.01 | gamma=100 | 62,6953 | ? | 0,000 | ? |
| Não | RBF | 100 | gamma=100 | 62,6953 | ? | 0,000 | ? |
| Sim | Poly | 1 | E=1 | 97,4609 | 0,989 | 0,942 | 0,965 |
| Sim | Poly | 0.01 | E=1 | 87,3047 | 1,000 | 0,660 | 0,795 |
| Sim | Poly | 100 | E=1 | 97,4609 | 0,968 | 0,963 | 0,966 |
| Sim | Poly | 1 | E=2 | 97,8516 | 0,984 | 0,958 | 0,971 |
| Sim | Poly | 0.01 | E=2 | 94,7266 | 1,000 | 0,859 | 0,924 |
| Sim | Poly | 100 | E=2 | 95,8984 | 0,947 | 0,942 | 0,945 |
| Sim | RBF | 1 | gamma=0.001 | 62,8906 | 1,000 | 0,005 | 0,010 |
| Sim | RBF | 0.01 | gamma=0.001 | 62,6953 | ? | 0,000 | ? |
| Sim | RBF | 100 | gamma=0.001 | 96,2891 | 1,000 | 0,901 | 0,948 |
| Sim | RBF | 1 | gamma=1 | 97,6563 | 0,979 | 0,958 | 0,968 |
| Sim | RBF | 0.01 | gamma=1 | 72,0703 | 1,000 | 0,251 | 0,402 |
| Sim | RBF | 100 | gamma=1 | 96,8750 | 0,963 | 0,953 | 0,958 |
| Sim | RBF | 1 | gamma=100 | 62,6953 | ? | 0,000 | ? |
| Sim | RBF | 0.01 | gamma=100 | 62,6953 | ? | 0,000 | ? |
| Sim | RBF | 100 | gamma=100 | 62,6953 | ? | 0,000 | ? |

`?` reproduz uma métrica indefinida na saída do Weka; não foi convertido em zero.

## Conclusões para a comparação

**Melhor por accuracy e F1 de M:** SMO normalizado, PolyKernel expoente 2, C=1: **97,8516%**, Precision **0,984**, Recall **0,958**, F1 **0,971**. Matriz: **[[183, 8], [3, 318]]**.
**Maior recall de M:** SMO normalizado, PolyKernel expoente 1, C=100: **0,963** (184/191 malignos identificados; 7 falsos negativos), com accuracy **97,4609%**, Precision **0,968** e F1 **0,966**. A escolha depende da métrica definida para o laboratório.
**Normalização:** mantendo C=1 e expoente 2, a accuracy passa de **92,1875%** para **97,8516%**, e o recall de M de **0,812** para **0,958**.
**Configurações que só preveem B:** algumas combinações RBF têm accuracy de **62,6953%**, recall de M igual a zero e Precision/F1 indefinidos. Isto é consistente com as matrizes; não constitui erro de transcrição.
**Resultados iguais:** as três configurações PolyKernel de expoente 2 sem normalização têm os mesmos resultados de validação; os valores de C nos relatórios são distintos e corretos.
Estes resultados selecionam o candidato SMO com os 90% de desenvolvimento. A comparação com KNN e a avaliação final nos 10% reservados pertencem às fases seguintes do laboratório.

SHA256 do treino: `9a2d2930ecb72eaf8320c33b1846f23b02d3aaeed2d95a1fd4da8ad1170d290f`
