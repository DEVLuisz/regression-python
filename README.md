# Projeto de Análise Preditiva em Python

Bem-vindo ao repositório de um projeto Python especializado em análise de dados e modelagem preditiva. Este projeto abrange a exploração, preparação e criação de modelos de regressão e classificação em uma base de dados contendo informações relevantes para prever salários.

## Funcionalidades e Etapas do Projeto

1. **Carregamento de Dados:** Utilizamos a biblioteca pandas para carregar dados a partir de um arquivo Excel (`Exame.xlsx`), contendo informações sobre variáveis independentes e a variável dependente 'salario'.

2. **Divisão de Dados:** A base é dividida em conjuntos de treino e teste (80% treino, 20% teste) usando a função `train_test_split` do scikit-learn.

3. **Escalonamento de Dados:** Os dados são padronizados para garantir consistência nos modelos por meio da classe `StandardScaler` do scikit-learn.

4. **Modelos de Classificação:** Construímos e treinamos modelos de classificação, incluindo Regressão Logística e Support Vector Classifier, para avaliar a previsão de categorias relacionadas aos salários.

5. **Modelos de Regressão:** Implementamos modelos de regressão, como Linear, PLS, Lasso, Ridge, Random Forest e Decision Tree, treinando-os para prever valores numéricos de salários.

6. **Avaliação de Desempenho:** Calculamos métricas de avaliação, como RMSE, R2 e Accuracy, para mensurar a precisão dos modelos tanto na previsão de valores quanto de categorias.

7. **Identificação do Melhor Modelo:** Identificamos o melhor modelo de regressão com base no menor RMSE, destacando a eficácia preditiva.

8. **Visualização de Resultados:** Utilizamos gráficos para comparar os valores reais e preditos na porção de teste, proporcionando insights visuais sobre o desempenho dos modelos.


## Estrutura do Projeto

- `main.py`: Script principal contendo o código de análise de dados e modelagem.
- `baseData/Exame.xlsx`: Arquivo Excel contendo a base de dados utilizada.

## Autor

Luís Felipe (Louís)

