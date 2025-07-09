import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/b9dd8e4b62b9e22ebcb9c8e89c271de4/raw/c69ec4b708fba03c445397b6a361db4345c83d7a/tracking.csv"


dados = pd.read_csv(uri)
exibicao = dados.head()
print(exibicao)

y = dados["comprou"]
y.head()

x = dados[["inicial", "palestras", "contato", "patrocinio"]]
x.head()

itens = dados.shape
print(itens)

treino_x = x[:75]
treino_y = y[:75]

teste_x = x[75:]
teste_y = y[75:]

print(f"Treinaremos com {len(treino_x)} elementos")
print(f"Testaremos com {len(teste_x)} elementos")


print("----------------")

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x,treino_y)
previsoes = modelo.predict(teste_x)


acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia: {acuracia:.2f}%")


print("----------------")

# Usando a biblioteca para separar treino e teste
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 325291 # numero aleatorio para ser sempre a mesma aleatoriedade


treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state= SEED, stratify=y)


print(f"Treinaremos com {len(treino_x)} elementos")
print(f"Testaremos com {len(teste_x)} elementos")


modelo = LinearSVC()
modelo.fit(treino_x,treino_y)
previsoes = modelo.predict(teste_x)


acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia: {acuracia:.2f}%")


quantidade1 = treino_y.value_counts()
quantidade2 = teste_y.value_counts()

print(quantidade1)
print(quantidade2)
