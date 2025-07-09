# Importação das bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Fase de carregamento dos dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/12291c548acaf544596795709020e3db/raw/325bdef098bd9cbc2189215b7e32e22f437f29f3/projetos.csv"
dados = pd.read_csv(uri)
exibicao = dados.head()
print(exibicao)

# ---------------------------------------------------------
# Fase de manipulação das features
dados["finalizado"] = dados["nao_finalizado"].map({1: 0, 0: 1})
dados.head()

# ---------------------------------------------------------
# Fase de visualização inicial
sns.relplot(x="horas_esperadas", y="preco", data=dados, hue="finalizado", col="finalizado")

# ---------------------------------------------------------
# Fase de limpeza dos dados (remoção de linhas inválidas)
dados = dados.query("horas_esperadas > 0")
dados.head()

plt.show()

print("----------------------")

# ---------------------------------------------------------
# Fase de separação de dados para treino e teste
x = dados[["horas_esperadas", "preco"]]
y = dados["finalizado"]
SEED = 20 # Garantir reprodutibilidade

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state=SEED, stratify=y)

print(f"Treinaremos com {len(treino_x)} elementos")
print(f"Testaremos com {len(teste_x)} elementos")

# ---------------------------------------------------------
# Fase de treino e avaliação com modelo LinearSVC
modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia: {acuracia:.2f}%")

# Comparando com modelo de base
base = teste_y.sum() / len(teste_y) * 100
print(f"A acurácia do modelo de base foi de {base:.2f}%")

print('----------------------')

# ---------------------------------------------------------
# Fase de visualização da decisão do LinearSVC
x_min = teste_x["horas_esperadas"].min()
x_max = teste_x["horas_esperadas"].max()
y_min = teste_x["preco"].min()
y_max = teste_x["preco"].max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)
xx, yy = np.meshgrid(eixo_x, eixo_y)

pontos = np.c_[xx.ravel(), yy.ravel()]
Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z)
plt.scatter(teste_x["horas_esperadas"], teste_x["preco"], c=teste_y, s=1)
plt.show()

print('------------------------------')

# ---------------------------------------------------------
# Fase de treino e avaliação com modelo SVC (não linear)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state=SEED, stratify=y)

print(f"Treinaremos com {len(treino_x)} elementos")
print(f"Testaremos com {len(teste_x)} elementos")

modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia: {acuracia:.2f}%")

# ---------------------------------------------------------
# Visualização da decisão do SVC (não normalizado)
x_min = teste_x["horas_esperadas"].min()
x_max = teste_x["horas_esperadas"].max()
y_min = teste_x["preco"].min()
y_max = teste_x["preco"].max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)
xx, yy = np.meshgrid(eixo_x, eixo_y)

pontos = np.c_[xx.ravel(), yy.ravel()]
Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z)
plt.scatter(teste_x["horas_esperadas"], teste_x["preco"], c=teste_y, s=1)
plt.show()

print('------------------------------')

# ---------------------------------------------------------
# Fase de treino e avaliação com modelo SVC + normalização
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, random_state=SEED, stratify=y)

print(f"Treinaremos com {len(raw_treino_x)} elementos")
print(f"Testaremos com {len(raw_teste_x)} elementos")

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia: {acuracia:.2f}%")

# ---------------------------------------------------------
# Visualização da decisão do SVC com dados normalizados
data_col1 = teste_x[:, 0]
data_col2 = teste_x[:, 1]

x_min = data_col1.min()
x_max = data_col1.max()
y_min = data_col2.min()
y_max = data_col2.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)
xx, yy = np.meshgrid(eixo_x, eixo_y)

pontos = np.c_[xx.ravel(), yy.ravel()]
Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z)
plt.scatter(data_col1, data_col2, c=teste_y, s=1)
plt.show()
