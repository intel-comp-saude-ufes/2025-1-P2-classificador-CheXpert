import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

# Tabela de desempenho dos 5 modelos em 10 dobras de validação cruzada
data = pd.DataFrame({
    'Modelo A': [0.85, 0.87, 0.84, 0.86, 0.85, 0.88, 0.84, 0.86, 0.85, 0.87],
    'Modelo B': [0.88, 0.89, 0.87, 0.89, 0.88, 0.91, 0.87, 0.89, 0.88, 0.90],
    'Modelo C': [0.90, 0.91, 0.89, 0.90, 0.90, 0.92, 0.89, 0.91, 0.90, 0.92],
    'Modelo D': [0.89, 0.90, 0.88, 0.89, 0.89, 0.91, 0.88, 0.90, 0.89, 0.91],
    'Modelo E': [0.84, 0.86, 0.83, 0.85, 0.84, 0.87, 0.83, 0.86, 0.84, 0.86]
})

print("Tabela de Desempenho dos Modelos:")
print(data)
print("-" * 50)

# Passo 1: Executando o Teste de Friedman
friedman_statistic, p_value = friedmanchisquare(
    data['Modelo A'],
    data['Modelo B'],
    data['Modelo C'],
    data['Modelo D'],
    data['Modelo E']
)

print("Resultados do Teste de Friedman:")
print(f"Estatística de Teste: {friedman_statistic:.4f}")
print(f"Valor p: {p_value:.4f}")
print("-" * 50)

# Passo 2: Verificando o valor p e executando o Teste de Nemenyi
alpha = 0.05

if p_value < alpha:
    print(f"O valor p ({p_value:.4f}) é menor que {alpha}.")
    print("Há uma diferença estatisticamente significativa entre os modelos.")
    print("Agora, vamos usar o Teste de Nemenyi para ver quais pares são diferentes.")
    print("-" * 50)

    # Executando o teste post-hoc de Nemenyi
    # A função espera os dados em um formato diferente, por isso usamos o .T (transposta)
    nemenyi_results = posthoc_nemenyi_friedman(data.T)

    print("Resultados do Teste de Nemenyi (Valores p para cada par):")
    print(nemenyi_results.round(4))

else:
    print(f"O valor p ({p_value:.4f}) é maior que {alpha}.")
    print("Não há diferença estatisticamente significativa entre os modelos.")