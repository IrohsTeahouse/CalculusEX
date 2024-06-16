import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter  # Adicionando importação do Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Necessário editar o caminho do csv
data = pd.read_csv('C:/Users/bleac/Desktop/Ex Garcia/DataSet/winequality-red.csv')

# Análise exploratória de dados
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.info())
print(data.describe())
print(data["quality"].unique())
print(Counter(data["quality"]))  # Usando Counter corretamente

# Gráfico de contagem de qualidade
sns.catplot(x="quality", data=data, kind="count")
plt.title('Distribuição da Qualidade do Vinho')
plt.xlabel('Qualidade')
plt.ylabel('Contagem')
plt.show()

# Matriz de correlação
correlation = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

# Boxplot de Acidez Volátil por Qualidade
plt.figure(figsize=(12, 6))
sns.boxplot(x='quality', y='volatile acidity', data=data)
plt.title('Boxplot de Acidez Volátil por Qualidade do Vinho')
plt.xlabel('Qualidade')
plt.ylabel('Acidez Volátil')
plt.show()

# Criação da variável "Reviews"
reviews = []
for i in data["quality"]:
    if i >= 1 and i <= 3:
        reviews.append("1")
    elif i >= 4 and i <= 7:
        reviews.append("2")
    elif i >= 8 and i <= 10:
        reviews.append("3")
data["Reviews"] = reviews

# Transformação das variáveis
x = data.iloc[:, :11]
y = data["Reviews"]

sc = StandardScaler()
x = sc.fit_transform(x)

# Divisão do conjunto de dados
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modelo de Regressão Logística
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)

# Métricas de avaliação para Regressão Logística
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("Logistic Regression Accuracy:", lr_acc_score)
print("Confusion Matrix:")
print(lr_conf_matrix)
