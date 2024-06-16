import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree=3):
        self.degree = degree
        self.b = None
        
    def fit(self, x, y):
        powers = self.__compute_powers(x)
        
        b1 = np.linalg.inv(np.dot(powers.T, powers))
        b2 = np.dot(powers.T, y)
        self.b = np.dot(b1, b2)
    
    def predict(self, x):
        if self.b is None:
            raise ValueError("Modelo não treinado. Por favor, ajuste o modelo antes de fazer previsões.")
        
        powers = self.__compute_powers(x)
        return np.dot(powers, self.b)
    
    def __compute_powers(self, x):
        x = x.ravel()
        powers = np.empty((x.shape[0], self.degree + 1))
        powers[:, 0] = np.ones(x.shape[0])
        powers[:, 1] = x
        
        for p in range(2, self.degree+1):
            powers[:, p] = x**p
        return powers

def generate_data(num_points, reg):
    x = np.linspace(1, 10, num_points)
    reg.fit(x.reshape(-1, 1), np.zeros_like(x))  # Treinando o modelo com dados fictícios
    
    # Usando a previsão do modelo para ajustar a função seno
    y_pred = reg.predict(x.reshape(-1, 1))
    
    # Adicionando um pequeno ruído
    y = y_pred + np.random.normal(0, 0.1, num_points)
    
    return x.reshape(-1, 1), y.reshape(-1, 1)

num_points = 100000
degree = 33

x, y = generate_data(num_points, PolynomialRegression(degree))

plt.scatter(x, y, s=3)  # Reduzindo o tamanho dos pontos

reg = PolynomialRegression(degree)
reg.fit(x, y)

y_pred = reg.predict(x)

plt.scatter(x, y, s=1)
plt.plot(x, y_pred, color='red')
plt.title(f'Polynomial Regression (Degree = {degree})')
plt.show()
