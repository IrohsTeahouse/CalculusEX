import numpy as np
import matplotlib.pyplot as plt

def gaussian_integral_monte_carlo(num_samples, a):
    # Função para integrar
    def integrand(x):
        return np.exp(-x**2)

    # Geração de amostras uniformemente distribuídas no intervalo [0, a]
    samples = np.random.uniform(0, a, num_samples)
    
    # Avaliação da função integrand para cada amostra
    function_values = integrand(samples)
    
    # Cálculo da média dos valores da função
    average_value = np.mean(function_values)
    
    # Estimativa da integral
    integral_estimate = a * average_value
    
    # Como a integral total é de -inf a +inf, multiplicamos por 2
    total_integral_estimate = 2 * integral_estimate
    
    return total_integral_estimate, samples, function_values

# Definições dos parâmetros
num_samples = 1000000  # Número de amostras
a = 5  # Limite superior do intervalo de integração

# Cálculo da integral
integral_estimate, samples, function_values = gaussian_integral_monte_carlo(num_samples, a)

print(f"Estimativa da Integral Gaussiana usando Monte Carlo: {integral_estimate}")

# Gerando o gráfico
x = np.linspace(0, a, 1000)
y = np.exp(-x**2)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Amostras')
plt.plot(x, y, 'r-', lw=2, label='$e^{-x^2}$')
plt.xlabel('x')
plt.ylabel('Densidade')
plt.title('Estimativa da Integral Gaussiana usando o Método de Monte Carlo')
plt.legend()
plt.show()