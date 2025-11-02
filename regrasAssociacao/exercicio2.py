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

base_mercado2 = pd.read_csv('mercado2.csv', header=None)

transacoes = []
for i in range(len(base_mercado2)):
    transacao = [str(base_mercado2.values[i, j]) for j in range(len(base_mercado2.columns))
                 if pd.notna(base_mercado2.values[i, j])]
    transacoes.append(transacao)

total_transacoes = len(transacoes)
print(f"Total de transações: {total_transacoes}")

from collections import Counter
todos_itens = []
for transacao in transacoes:
    todos_itens.extend(transacao)

# a) Produtos vendidos pelo menos 28 vezes por semana (4 vezes/dia * 7 dias)
min_vendas = 28
min_support = min_vendas / total_transacoes

print(f"\nProdutos que aparecem pelo menos {min_vendas} vezes")
print(f"Suporte mínimo calculado: {min_support:.4f} ({min_support*100:.2f}%)")
print(f"Fórmula: {min_vendas} / {total_transacoes} = {min_support:.4f}")



regras_a = apriori(
    transacoes,
    min_support=min_support,
    min_confidence=0.01,
    min_lift=1,
    min_length=2
)
resultados_a = list(regras_a)
print(f"\n(a) Qtde de registros (RelationRecords) com suporte >= {min_support:.4f}: {len(resultados_a)}")

regras_b = apriori(
    transacoes,
    min_support=min_support,
    min_confidence=0.2,
    min_lift=3,
    min_length=2
)
resultados_b = list(regras_b)
print(f"\n(b) Qtde de registros com conf>=0.2 e lift>=3: {len(resultados_b)}")

resultados = resultados_b  # <-- escolhemos visualizar as regras filtradas de (b)

A = []          # lista para armazenar antecedentes (A) de cada regra
B = []          # lista para armazenar consequentes (B) de cada regra
suporte = []    # lista para armazenar o suporte do conjunto de itens da regra
confianca = []  # lista para armazenar a confiança da regra
lift = []       # lista para armazenar o lift da regra

for resultado in resultados:
  s = resultado[1]          # suporte do conjunto (RelationRecord.support)
  result_rules = resultado[2]  # lista de OrderedStatistics (regras A -> B derivadas desse conjunto)
  for result_rule in result_rules:
    a = list(result_rule[0])  # antecedente (items_base) convertido em lista
    b = list(result_rule[1])  # consequente (items_add) convertido em lista
    c = result_rule[2]        # confiança da regra
    l = result_rule[3]        # lift da regra
    A.append(a)               # acumula antecedente
    B.append(b)               # acumula consequente
    suporte.append(s)         # acumula suporte (do conjunto de itens)
    confianca.append(c)       # acumula confiança
    lift.append(l)            # acumula lift

rules_df = pd.DataFrame({'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})
print("\n(c) Top 10 regras por lift:")
print(rules_df.sort_values(by='lift', ascending=False).head(10))

# ===========================
# (d) Ordene por confiança e mostre a maior confiança
# ===========================
rules_conf = rules_df.sort_values(by='confianca', ascending=False)
print("\n(d) Top 10 regras por confiança:")
print(rules_conf.head(10))

if not rules_conf.empty:
    print(f"\n(d) Maior confiança encontrada: {rules_conf['confianca'].iloc[0]:.4f}")
else:
    print("\n(d) Nenhuma regra encontrada para os parâmetros definidos.")


