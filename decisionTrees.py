# Importação das bibliotecas
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import graphviz
# ---------------------------------------------------------
# Carregamento e preparação dos dados
dados = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/dd7ba8142321c2c8aaa0ddd6c8862fcc/raw/e694a9b43bae4d52b6c990a5654a193c3f870750/precos.csv")

# Criando coluna km_por_ano a partir de milhas_por_ano
dados["km_por_ano"] = dados["milhas_por_ano"] * 1.60934

# Criando coluna idade a partir do ano atual
dados["idade"] = datetime.today().year - dados["ano_do_modelo"]

# Removendo colunas
dados.drop(["milhas_por_ano", "ano_do_modelo"], axis=1, inplace=True)

# Exibindo as primeiras linhas do dataframe para conferência
print(dados.head())

# ---------------------------------------------------------
# Separação das features (X) e do target (y)
x = dados[["preco", "idade", "km_por_ano"]]
y = dados["vendido"]

# Divisão entre treino e teste
SEED = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(
    x, y, random_state=SEED, stratify=y
)

print(f"Treinaremos com {len(treino_x)} elementos")
print(f"Testaremos com {len(teste_x)} elementos")

# ---------------------------------------------------------
# Normalização dos dados
# Isso ajuda o modelo SVM a ter um desempenho melhor,
# pois ele funciona melhor com dados na mesma escala
# scaler = StandardScaler()
# scaler.fit(raw_treino_x)
# treino_x = scaler.transform(raw_treino_x)
# teste_x = scaler.transform(raw_teste_x)

# ---------------------------------------------------------
# Modelo de Árvore de decisão
modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(treino_x, treino_y)

# Fazendo previsões com os dados de teste
previsoes = modelo.predict(teste_x)

# Calculando acurácia: percentual de previsões corretas
acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia do modelo de Árvore de decisão: {acuracia:.2f}%")


estrutura = export_graphviz(modelo, filled=True, rounded=True, feature_names=x.columns, class_names=["não", "sim"])
grafico = graphviz.Source(estrutura)

grafico.render("arvore_de_decisao", format="pdf", cleanup=True)