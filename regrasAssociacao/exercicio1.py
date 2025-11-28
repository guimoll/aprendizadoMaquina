#Guilherme moll
import pandas as pd

from apyori import apriori

def exibir_regras(resultados):
    for i, regra in enumerate(resultados, 1):
        print(f"\nRegra {i}:")
        print(f"  Items: {list(regra.items)}")
        print(f"  Support: {regra.support:.4f}")
        for stat in regra.ordered_statistics:
            print(f"  {list(stat.items_base)} → {list(stat.items_add)}")
            print(f"  Confidence: {stat.confidence:.4f}, Lift: {stat.lift:.4f}")

def exibir_formatado(resultados_formatados):
    for i, regra in enumerate(resultados_formatados, 1):
        print(f"\nRegra Formatada {i}:")
        items = [''.join(item) if isinstance(item, list) else item for item in regra]
        print(f"  Items: {items}")
        print()

base_mercado = pd.read_csv('mercado.csv', header = None)

transacoes = []
for i in range(0, len(base_mercado)):
    transacoes.append([str(base_mercado.values[i,j]) for j in range(0, 4)])

regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2, min_length = 2)

resultados = list(regras)
exibir_regras(resultados)

resultados2 = [list(x) for x in resultados]
resultadoFormatado = []
for j in range(0,3):
    resultadoFormatado.append([list(x) for x in resultados2[j][0]])

exibir_formatado(resultadoFormatado)
