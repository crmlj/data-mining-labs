# Dados e atribuição

Para reproduzir os resultados com as mesmas partições, copia os quatro ficheiros ARFF do teu arquivo dos laboratórios para os caminhos indicados abaixo. Os hashes SHA-256 permitem confirmar que são os ficheiros usados nas experiências. As fontes originais estão disponíveis na UCI, que indica a licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) para os dois datasets.

| Laboratório | Fonte original | Preparação das partições originais |
| --- | --- | --- |
| 7 | Wolberg, W., Mangasarian, O., Street, N. e Street, W. (1993). [Breast Cancer Wisconsin (Diagnostic)](https://doi.org/10.24432/C5DW2B). UCI Machine Learning Repository. | Conversão para ARFF, remoção do identificador, classe na primeira coluna e divisão estratificada em 512 instâncias de treino e 57 de teste. |
| 8 | Sigillito, V., Wing, S., Hutton, L. e Baker, K. (1989). [Ionosphere](https://doi.org/10.24432/C5W01B). UCI Machine Learning Repository. | A versão em ARFF usada na disciplina foi dividida com `StratifiedRemoveFolds` no Weka. A partição 2 de 5, com seed 1, ficou para teste. O complemento ficou para treino. |

No Lab 8, uma instância `bad` aparece com os mesmos atributos nos dois conjuntos. A avaliação original usou essas partições tal como foram guardadas.

## SHA-256 das partições

- `9a2d2930ecb72eaf8320c33b1846f23b02d3aaeed2d95a1fd4da8ad1170d290f` — `Lab7/wdbc-train.arff`
- `f29b24ec333048844844bb542b789121ebdd5d7dbb5693b9769c9366d1231e0f` — `Lab7/wdbc-test.arff`
- `864870c394d824a90a7e1ff10cbaac1ffeba450f008e913641153e9f0821f927` — `Lab8/train_ionosphere.arff`
- `b98b5c4ca844d09970fbf7526117ed4a6ffa09f248033b7669ccbced403caa68` — `Lab8/test_ionosphere.arff`
