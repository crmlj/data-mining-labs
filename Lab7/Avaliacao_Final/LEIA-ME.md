# Lab 7 — 7.3: seleção do modelo e avaliação independente

## Comparação na validação cruzada

Treino: 512 instâncias. Validação cruzada estratificada de 10 folds, seed 1. Classe positiva: M.
Critério de seleção: maior F1 de M, usando Accuracy como desempate. O vencedor também apresenta a maior Accuracy.

| Modelo candidato | Accuracy (%) | Precision M | Recall M | F1 M |
|---|---:|---:|---:|---:|
| SMO: Normalize, PolyKernel E=2, C=1 | 97,8516 | 0,984 | 0,958 | 0,971 |
| KNN / IBk: k=9 | 97,0703 | 0,989 | 0,932 | 0,960 |

**Modelo selecionado: SMO, C=1, PolyKernel expoente 2, normalização ativa (-N 0).**
A escolha foi registada antes da avaliação do teste independente. Não houve pesquisa de parâmetros no teste.
O modelo foi treinado com todos os 512 exemplos do treino. A normalização interna do SMO foi ajustada no treino.

## Validação cruzada versus teste independente

| Avaliação | Instâncias | Accuracy (%) | Precision M | Recall M | F1 M |
|---|---:|---:|---:|---:|---:|
| Validação cruzada no treino | 512 | 97,8516 | 0,984 | 0,958 | 0,971 |
| Teste independente | 57 | 98,2456 | 1,000 | 0,952 | 0,976 |

Diferença de Accuracy (teste menos CV): **0,3941 pontos percentuais**.
Diferenças das métricas de M (teste menos CV): Precision 0,016, Recall -0,006, F1 0,005.

## Matriz de confusão do teste

Classes reais nas linhas e previstas nas colunas:

| Real / Previsto | M | B |
|---|---:|---:|
| M | 20 | 1 |
| B | 0 | 36 |

Acertos: 56/57. Falsos negativos: 1. Falsos positivos: 0.

## Interpretação e verificação

A validação cruzada estima o desempenho durante a seleção; o teste avalia o modelo final em dados reservados.
O teste pode apresentar resultados superiores ou inferiores aos da CV por variação da amostra.
Com apenas 57 exemplos, cada erro altera a Accuracy em cerca de 1,75 pontos percentuais.
Os resultados não permitem afirmar que um aumento no teste prova superioridade ou ausência de sobreajuste.
Não se alteraram parâmetros após observar o teste. Accuracy e métricas de ambas as classes conferem com a matriz.
Os conjuntos têm atributos compatíveis e não partilham linhas. O modelo guardado foi recarregado e reproduziu a avaliação.

## Ficheiros

- SMO_norm_Poly_E2_C1_teste_independente.txt: saída completa do Weka e comando.
- SMO_norm_Poly_E2_C1.model: modelo criado pelo script de reprodução.
- selecao_por_validacao_cruzada.json: decisão e candidatos registados antes do teste.
- resultado_final.json: métricas, diferenças, hashes e verificações.
- executar_avaliacao_final.py: reprodução com o motor Java do Weka.

Treino SHA256: `9a2d2930ecb72eaf8320c33b1846f23b02d3aaeed2d95a1fd4da8ad1170d290f`
Teste SHA256: `f29b24ec333048844844bb542b789121ebdd5d7dbb5693b9769c9366d1231e0f`
