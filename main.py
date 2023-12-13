import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Carrega a base de dados que estava no excel
data = pd.read_excel('./baseData/Exame.xlsx')

# Define 'salario' como variável dependente e as outras como preditoras
X = data.drop('salario', axis=1)
y = data['salario']

# Divide a base em porção de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalonamento de dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cria modelos de classificação
classification_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Support Vector Classifier': SVC()
}

for model_name, model in classification_models.items():
    model.fit(X_train_scaled, y_train)

# Aplica os modelos de regressão na porção de teste e calcula métricas de avaliação
regression_models = {
    'Linear Regression': LinearRegression(),
    'PLS Regression': PLSRegression(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor()
}

results = {'Model': [], 'RMSE': [], 'R2': [], 'MAPE': []}
predictions = {}

for model_name, model in regression_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_error(y_test, y_pred) / abs(y_test.mean()) * 100

    results['Model'].append(model_name)
    results['RMSE'].append(rmse)
    results['R2'].append(r2)
    results['MAPE'].append(mape)

    predictions[model_name] = y_pred

# Mostra a tabela das métricas para os modelos de regressão
results_df = pd.DataFrame(results)
print("Métricas para Modelos de Regressão:")
print(results_df)

# Avalia os modelos de classificação na porção de teste
classification_results = {'Model': [], 'Accuracy': []}

for model_name, model in classification_models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    classification_results['Model'].append(model_name)
    classification_results['Accuracy'].append(accuracy)

# Mostra a tabela das métricas para os modelos de classificação
classification_results_df = pd.DataFrame(classification_results)
print("\nMétricas para Modelos de Classificação:")
print(classification_results_df)

# Identifica o melhor modelo com base em uma métrica específica (por exemplo, menor RMSE)
melhor_modelo = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
print(f'\nMelhor Modelo de Regressão (Menor RMSE): {melhor_modelo}')

# Desenha um gráfico comparando os valores reais e preditos na porção de teste
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Real', marker='o')

for model_name, y_pred in predictions.items():
    plt.plot(y_pred, label=model_name)

plt.xlabel('Amostras')
plt.ylabel('Valor')
plt.legend()
plt.title('Comparação de Modelos na Porção de Teste')
plt.show()
