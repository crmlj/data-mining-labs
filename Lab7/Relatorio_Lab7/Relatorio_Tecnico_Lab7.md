# Comparação de métodos de classificação no conjunto WDBC

Carlos Lima | Laboratório 7 de Data Mining

## 1 Tabela de resultados da validação cruzada

Compararam-se ZeroR, OneR, k-NN e SVM no Weka 3.8.7, usando o WDBC, com 569 instâncias e 30 preditores numéricos. Removeu-se o identificador e definiu-se Diagnosis como classe: M (maligna) positiva e B (benigna) negativa. A divisão estratificada reservou 512 exemplos para treino (191 M e 321 B) e 57 para teste independente (21 M e 36 B).

As Tabelas 1 e 2 apresentam os 42 resultados de validação cruzada de 10 partições no treino: ZeroR, OneR, dez valores de k e 30 configurações SMO. Os ensaios parametrizados usaram a semente de avaliação 1. Precisão, sensibilidade e F1 referem-se à classe M.

Tabela 1 Resultados completos de ZeroR OneR e k-NN

| Algoritmo | k | Exatidão (%) | Precisão M | Sensibilidade M | F1 M |
| --- | --- | --- | --- | --- | --- |
| ZeroR | - | 62,6953 | n.d. | 0,000 | n.d. |
| OneR | - | 88,8672 | 0,881 | 0,812 | 0,845 |
| k-NN (IBk) | 1 | 94,5313 | 0,931 | 0,921 | 0,926 |
| k-NN (IBk) | 3 | 96,6797 | 0,978 | 0,932 | 0,954 |
| k-NN (IBk) | 5 | 96,8750 | 0,983 | 0,932 | 0,957 |
| k-NN (IBk) | 7 | 96,8750 | 0,978 | 0,937 | 0,957 |
| k-NN (IBk) | 9 | 97,0703 | 0,989 | 0,932 | 0,960 |
| k-NN (IBk) | 11 | 96,8750 | 0,994 | 0,921 | 0,957 |
| k-NN (IBk) | 15 | 96,4844 | 0,989 | 0,916 | 0,951 |
| k-NN (IBk) | 19 | 96,6797 | 0,994 | 0,916 | 0,954 |
| k-NN (IBk) | 25 | 95,7031 | 0,977 | 0,906 | 0,940 |
| k-NN (IBk) | 31 | 95,7031 | 0,988 | 0,895 | 0,940 |

ZeroR não tem hiperparâmetros ajustados. No OneR manteve-se minBucketSize = 6. A indicação n.d. reproduz uma métrica não definida pelo Weka: ZeroR não previu qualquer caso M. No IBk mantiveram-se a distância Euclidiana normalizada e as restantes opções; a melhor configuração por F1 foi k = 9, assinalada na tabela.

Na Tabela 2, E designa o expoente do PolyKernel e g designa gamma do RBF. A grelha cruza três valores de C com dois expoentes ou três valores de gamma, com e sem normalização. A linha assinalada identifica o SMO selecionado: C = 1, PolyKernel de expoente 2, com normalização.



Tabela 2 Resultados completos das configurações SVM executadas pelo SMO

| Normalização | Kernel | C | Expoente / gamma | Exatidão (%) | Precisão M | Sensibilidade M | F1 M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sem | PolyKernel | 0,01 | E = 1 | 94,9219 | 0,941 | 0,921 | 0,931 |
| Sem | PolyKernel | 1 | E = 1 | 95,3125 | 0,956 | 0,916 | 0,936 |
| Sem | PolyKernel | 100 | E = 1 | 96,6797 | 0,958 | 0,953 | 0,955 |
| Sem | PolyKernel | 0,01 | E = 2 | 92,1875 | 0,975 | 0,812 | 0,886 |
| Sem | PolyKernel | 1 | E = 2 | 92,1875 | 0,975 | 0,812 | 0,886 |
| Sem | PolyKernel | 100 | E = 2 | 92,1875 | 0,975 | 0,812 | 0,886 |
| Sem | RBF | 0,01 | g = 0,001 | 62,6953 | n.d. | 0,000 | n.d. |
| Sem | RBF | 1 | g = 0,001 | 92,9688 | 0,897 | 0,916 | 0,907 |
| Sem | RBF | 100 | g = 0,001 | 91,2109 | 0,869 | 0,901 | 0,884 |
| Sem | RBF | 0,01 | g = 1 | 62,6953 | n.d. | 0,000 | n.d. |
| Sem | RBF | 1 | g = 1 | 62,6953 | n.d. | 0,000 | n.d. |
| Sem | RBF | 100 | g = 1 | 62,6953 | n.d. | 0,000 | n.d. |
| Sem | RBF | 0,01 | g = 100 | 62,6953 | n.d. | 0,000 | n.d. |
| Sem | RBF | 1 | g = 100 | 62,6953 | n.d. | 0,000 | n.d. |
| Sem | RBF | 100 | g = 100 | 62,6953 | n.d. | 0,000 | n.d. |
| Com | PolyKernel | 0,01 | E = 1 | 87,3047 | 1,000 | 0,660 | 0,795 |
| Com | PolyKernel | 1 | E = 1 | 97,4609 | 0,989 | 0,942 | 0,965 |
| Com | PolyKernel | 100 | E = 1 | 97,4609 | 0,968 | 0,963 | 0,966 |
| Com | PolyKernel | 0,01 | E = 2 | 94,7266 | 1,000 | 0,859 | 0,924 |
| Com | PolyKernel | 1 | E = 2 | 97,8516 | 0,984 | 0,958 | 0,971 |
| Com | PolyKernel | 100 | E = 2 | 95,8984 | 0,947 | 0,942 | 0,945 |
| Com | RBF | 0,01 | g = 0,001 | 62,6953 | n.d. | 0,000 | n.d. |
| Com | RBF | 1 | g = 0,001 | 62,8906 | 1,000 | 0,005 | 0,010 |
| Com | RBF | 100 | g = 0,001 | 96,2891 | 1,000 | 0,901 | 0,948 |
| Com | RBF | 0,01 | g = 1 | 72,0703 | 1,000 | 0,251 | 0,402 |
| Com | RBF | 1 | g = 1 | 97,6563 | 0,979 | 0,958 | 0,968 |
| Com | RBF | 100 | g = 1 | 96,8750 | 0,963 | 0,953 | 0,958 |
| Com | RBF | 0,01 | g = 100 | 62,6953 | n.d. | 0,000 | n.d. |
| Com | RBF | 1 | g = 100 | 62,6953 | n.d. | 0,000 | n.d. |
| Com | RBF | 100 | g = 100 | 62,6953 | n.d. | 0,000 | n.d. |

## 2 Procedimento de seleção do modelo

A separação inicial usou StratifiedRemoveFolds, com 10 partições, a primeira partição e a semente 1. A seleção invertida originou o treino e a seleção complementar originou o teste, preservando aproximadamente a proporção de malignos: 37,30% e 36,84%, respetivamente. O teste permaneceu reservado durante a seleção.

Cada configuração foi avaliada em dez iterações: nove partições para aprendizagem e uma para validação, agregando-se as previsões. No IBk variou-se apenas k, mantendo LinearNNSearch, distância Euclidiana com normalização interna, ausência de ponderação por distância e seleção automática de k desativada. No SMO cruzaram-se os valores de C, kernel e expoente ou gamma com os dois modos de normalização. A transformação foi ajustada pelo SMO no treino de cada avaliação. As métricas foram conferidas com as matrizes de confusão.

O critério principal foi o maior F1 de M, com a exatidão como desempate. Examinaram-se também a sensibilidade e os erros por classe; não se definiu uma matriz de custos. O SMO selecionado apresentou simultaneamente o maior F1 (0,971) e a maior exatidão (97,8516%), face a 0,960 e 97,0703% no melhor IBk, k = 9. Na validação, o SMO produziu oito falsos negativos e três falsos positivos; o IBk produziu treze e dois. A escolha reduziu cinco casos M não identificados, com mais um falso positivo.

O SMO normalizado com kernel linear e C = 100 alcançou maior sensibilidade (0,963), mas menor F1 (0,966). Foi mantida a configuração de expoente 2 por cumprir melhor o critério definido. A decisão foi registada antes da avaliação independente.

## 3 Avaliação final no conjunto de dados de teste independente

Treinou-se o SMO selecionado com os 512 exemplos do treino, usando C = 1, PolyKernel de expoente 2 e normalização ativa. A transformação aprendida no treino foi aplicada ao teste. Avaliaram-se exclusivamente os 57 exemplos independentes, sem reajustar os hiperparâmetros.

Tabela 3 Resultados no teste independente

| Exatidão (%) | Precisão M | Sensibilidade M | F1 M |
| --- | --- | --- | --- |
| 98,2456 | 1,000 | 0,952 | 0,976 |

Tabela 4 Matriz de confusão do teste independente

| Classe real / Classe prevista | M | B | Total |
| --- | --- | --- | --- |
| M | 20 | 1 | 21 |
| B | 0 | 36 | 36 |
| Total | 20 | 37 | 57 |

Obtiveram-se 56 acertos em 57 casos: 20 verdadeiros positivos, 36 verdadeiros negativos, um falso negativo e nenhum falso positivo. Foram identificados 20 dos 21 malignos. A exatidão (accuracy) foi 98,2456%; a precisão (precision), 1,000; a sensibilidade (recall), 0,952; e a medida F1, 0,976.

## 4 Discussão crítica dos resultados

Os três gráficos apresentados no ficheiro complementar Graficos_Lab7.pdf foram gerados com o apoio das ferramentas da OpenAI, com o propósito de mapear e compreender com maior clareza os efeitos da variação dos hiperparâmetros. Incluem as variações de k, a exatidão das configurações SMO e o recall dos casos malignos.

No k-NN, a relação com k não foi monotónica: a exatidão passou de 94,5313% em k = 1 para 97,0703% em k = 9, oscilou nos valores seguintes e terminou em 95,7031% para k = 25 e 31. O melhor F1 ocorreu em k = 9, mas a maior sensibilidade (0,937) ocorreu em k = 7. Aumentar o número de vizinhos não melhorou sistematicamente a classificação.

No SVM, C maior penaliza mais os erros no treino e reduz a regularização. Com PolyKernel de expoente 2 e normalização, C = 0,01, 1 e 100 produziram exatidões de 94,7266%, 97,8516% e 95,8984%. O desempenho também não foi monotónico. No RBF, gamma pequeno amplia a influência de cada exemplo e gamma elevado torna-a mais localizada. Com C = 1 e normalização, gamma = 0,001, 1 e 100 deram 62,8906%, 97,6563% e 62,6953%. Com gamma = 100, o modelo previu apenas B e teve sensibilidade nula.

A normalização alterou substancialmente o SVM. Com C = 1 e PolyKernel de expoente 2, a exatidão aumentou de 92,1875% para 97,8516%, colocando os atributos em escalas comparáveis. O efeito depende da configuração: com kernel linear e C = 0,01, a exatidão desceu de 94,9219% para 87,3047%. No k-NN, a normalização interna da distância manteve-se ativa em todos os ensaios, pelo que o seu efeito isolado não foi medido. ZeroR e OneR mantiveram as configurações de referência.

No modelo selecionado, o teste superou a exatidão da validação cruzada em 0,3941 pontos percentuais (98,2456% contra 97,8516%). A precisão aumentou de 0,984 para 1,000 e o F1 de 0,971 para 0,976; a sensibilidade desceu de 0,958 para 0,952. O teste é pequeno: um erro representa 1,75 pontos percentuais de exatidão e 4,76 pontos percentuais de sensibilidade entre os 21 malignos. A escolha entre várias configurações pode tornar a validação otimista; a avaliação independente limita esse efeito, mas uma única separação não permite concluir que diferenças pequenas sejam estatisticamente significativas.

Um falso negativo classifica uma amostra maligna como benigna e poderia atrasar a identificação de doença num contexto de apoio ao diagnóstico. Um falso positivo classifica uma amostra benigna como maligna e poderia motivar preocupação e investigação adicional. O impacto potencial dos falsos negativos exige atenção à sensibilidade: o único erro no teste foi desse tipo. A ausência de falsos positivos nos 36 benignos observados não garante a sua ausência noutras amostras. O modelo oferece o melhor compromisso segundo o F1 e a exatidão; uma seleção orientada por custos clínicos exigiria definir esses custos previamente e validar o desempenho em dados adicionais.
