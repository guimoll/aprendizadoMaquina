import matUtils
import graficos
# Guilherme Moll, Gabriel Ramos, Eduardo brandt


#=== ANÁLISE ESTATÍSTICA DOS DADOS ===
#              area    quartos          preco
#count    47.000000  47.000000      47.000000
#mean   2000.680851   3.170213  340412.765957
#std     794.702354   0.760982  125039.911223
#min     852.000000   1.000000  169900.000000
#25%    1432.000000   3.000000  249900.000000
#50%    1888.000000   3.000000  299900.000000
#75%    2269.000000   4.000000  384450.000000
#max    4478.000000   5.000000  699900.000000

#ITEM A: Média de preço das casas: $340,412.77
#ITEM B: Menor casa custa: $169,900.00
#   - Área: 1000.0 sq ft
#   - Quartos: 1.0
# Casa mais cara custa: $699,900.00
#   - Área: 4478.0 sq ft
#   - Quartos: 5.0

#ITEM C:
#x,y = matUtils.load_data()
#print(matUtils.load_data())

#ITEM D:
#x,y = matUtils.load_data_quartos_preco()
#graficos.plot_dataset(x, y, "PLOT Numero de quartos e preço")
#z,v = matUtils.load_data_tamanho_casa_preco()
#graficos.plot_dataset(z, v, "PLOT Tamanho casa e preço")

#ITEM E
#print(matUtils.regressao_multipla())
#b0,b1,b2 = matUtils.regressao_multipla()
#graficos.plot_regression_3d(b0,b1,b2)
# Testar também com a escala normalizada (use_scaling=True)

#Item F,G
#X, Y = matUtils.load_data_tamanho_casa_preco()
#graficos.plot_dataset_regressao_3d(X, Y, "PLOT 3D Tamanho casa e preço")
#Z,V = matUtils.load_data_quartos_preco()
#graficos.plot_dataset_regressao_3d(Z, V, "Plot 3D Numero de quartos e preço")

#Item H
#b0, b1, b2 = matUtils.regressao_multipla()
#tamanho = 1650
#quartos = 4
#preco_previsto = b0 + b1 * tamanho + b2 * quartos
#print(f"Preço previsto para {tamanho} sqft e {quartos} quartos: {preco_previsto:.2f}")
#231916 com 10 quartos
#293081 com 3 quartos
#310557 com 1 quarto
#Isso ocorre pois o coeficiente b2 é negativo, ou seja, quanto mais quartos, menor o preço vai ser


#Item i
#print("Preço vindo das libs:", matUtils.regressao_multipla_libs())
#print("Preço vindo do método manual:", matUtils.regressao_multipla())

def item_a():
    return "Média de preço das casas: $340,412.77"

def item_b():
    return ("Menor casa custa: $169,900.00\n"
            "- Área: 1000.0 sq ft\n"
            "- Quartos: 1.0\n"
            "Casa mais cara custa: $699,900.00\n"
            "- Área: 4478.0 sq ft\n"
            "- Quartos: 5.0")

def item_c():
    x, y = matUtils.load_data()
    return f"Dados carregados: {len(x)} casas"

def item_d():
    x, y = matUtils.load_data_quartos_preco()
    graficos.plot_dataset(x, y, "PLOT Numero de quartos e preço")
    z, v = matUtils.load_data_tamanho_casa_preco()
    graficos.plot_dataset(z, v, "PLOT Tamanho casa e preço")
    return "Plots gerados para quartos/preço e tamanho/preço."

def item_e():
    b0, b1, b2 = matUtils.regressao_multipla()
    graficos.plot_regression_3d(b0, b1, b2)
    return f"Regressão múltipla: b0={b0:.2f}, b1={b1:.2f}, b2={b2:.2f}"

def item_f():
    X, Y = matUtils.load_data_tamanho_casa_preco()
    graficos.plot_dataset_regressao_3d(X, Y, "PLOT 3D Tamanho casa e preço")
    return "Plot 3D Tamanho casa e preço gerado."

def item_g():
    Z, V = matUtils.load_data_quartos_preco()
    graficos.plot_dataset_regressao_3d(Z, V, "Plot 3D Numero de quartos e preço")
    return "Plot 3D Numero de quartos e preço gerado."

def item_h(tamanho=1650, quartos=4):
    b0, b1, b2 = matUtils.regressao_multipla()
    preco_previsto = b0 + b1 * tamanho + b2 * quartos
    return (f"Preço previsto para {tamanho} sqft e {quartos} quartos: {preco_previsto:.2f}\n"
            "231916 com 10 quartos\n293081 com 3 quartos\n310557 com 1 quarto\n"
            "Coeficiente b2 negativo: mais quartos, menor preço.")

def item_i():
    preco_libs = matUtils.regressao_multipla_libs()
    preco_manual = matUtils.regressao_multipla()
    return (f"Preço vindo das libs: {preco_libs}\n"
            f"Preço vindo do método manual: {preco_manual}")
