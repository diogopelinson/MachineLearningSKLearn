# ---------------------------------------------------------
# Importação das bibliotecas
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

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

# ---------------------------------------------------------
# Divisão entre treino e teste
SEED = 20
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(
    x, y, random_state=SEED, stratify=y
)

print(f"Treinaremos com {len(raw_treino_x)} elementos")
print(f"Testaremos com {len(raw_teste_x)} elementos")

# ---------------------------------------------------------
# Normalização dos dados
# Isso ajuda o modelo SVM a ter um desempenho melhor,
# pois ele funciona melhor com dados na mesma escala
scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

# ---------------------------------------------------------
# Treinamento e avaliação do modelo SVM
modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)

# Fazendo previsões com os dados de teste
previsoes = modelo.predict(teste_x)

# Calculando acurácia: percentual de previsões corretas
acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia do modelo SVM: {acuracia:.2f}%")

# ---------------------------------------------------------
# DummyClassifier - cria um modelo "de base"
# Serve para comparação: é um modelo simples que "chuta"
# Usamos strategy='stratified' para ele manter a mesma
# proporção de classes do treino
SEED = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state=SEED, stratify=y)

print(f"Treinaremos com {len(treino_x)} elementos")
print(f"Testaremos com {len(teste_x)} elementos")

# Criando e treinando o Dummy
classificador = DummyClassifier(strategy='stratified')
classificador.fit(treino_x, treino_y)

# Fazendo previsões e calculando acurácia
previsoes = classificador.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia do Dummy foi de: {acuracia:.2f}%")
