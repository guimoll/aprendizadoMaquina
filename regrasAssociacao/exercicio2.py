import pandas as pd
from apyori import apriori

def exibir_regras(resultados):
    if not resultados:
        print("Nenhuma regra encontrada com os parâmetros especificados.")
        return

    for i, regra in enumerate(resultados, 1):
        print(f"\nRegra {i}:")
        print(f"  Items: {list(regra.items)}")
        print(f"  Support: {regra.support:.4f}")
        for stat in regra.ordered_statistics:
            print(f"  {list(stat.items_base)} → {list(stat.items_add)}")
            print(f"  Confidence: {stat.confidence:.4f}, Lift: {stat.lift:.4f}")

# Carregar a base de dados mercado2
base_mercado2 = pd.read_csv('mercado2.csv', header=None)

# Preparar transações
transacoes = []
for i in range(len(base_mercado2)):
    # Pegar todos os itens não-nulos de cada linha
    transacao = [str(base_mercado2.values[i, j]) for j in range(len(base_mercado2.columns))
                 if pd.notna(base_mercado2.values[i, j])]
    transacoes.append(transacao)

# Calcular o total de transações
total_transacoes = len(transacoes)
print(f"Total de transações: {total_transacoes}")

# Calcular o total de produtos ÚNICOS/DISTINTOS na base de dados
from collections import Counter
todos_itens = []
for transacao in transacoes:
    todos_itens.extend(transacao)

# Produtos únicos/distintos
produtos_unicos = set(todos_itens)
total_produtos = len(produtos_unicos)
print(f"Total de produtos únicos/distintos na base de dados: {total_produtos}")

# a) Produtos vendidos pelo menos 28 vezes por semana (4 vezes/dia * 7 dias)
# Suporte mínimo = 28 / total_produtos (produtos únicos)
min_vendas = 28
min_support = min_vendas / total_produtos

print(f"\nProdutos que aparecem pelo menos {min_vendas} vezes")
print(f"Suporte mínimo calculado: {min_support:.4f} ({min_support*100:.2f}%)")
print(f"Fórmula: {min_vendas} / {total_produtos} = {min_support:.4f}")

