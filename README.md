# 🧠 Machine Learning com Scikit-Learn — Classificadores e Visualização

Este repositório contém diversos experimentos de aprendizado de máquina utilizando a biblioteca **Scikit-Learn**. Os dados são públicos e foram extraídos de gists para projetos didáticos, com foco em **classificação supervisionada**, avaliação de modelos e visualizações de decisão.

---

## 📊 Modelos Utilizados

### 🌳 Decision Tree Classifier
- **Classe:** `DecisionTreeClassifier(max_depth=3)`
- **Uso:** Previsão de venda de carros com base em preço, idade e km por ano.
- **Destaque:** Geração de árvore de decisão com `graphviz`.

---

### 🔍 LinearSVC
- **Classe:** `LinearSVC()`
- **Usos:**
  - Classificação de animais com base em atributos físicos.
  - Previsão de conversão de leads.
  - Classificação de projetos como finalizados ou não.

---

### 📈 SVC (Support Vector Classifier)
- **Classe:** `SVC(gamma='auto')`
- **Usos:**
  - Classificação de projetos com e sem normalização (`StandardScaler`).
  - Visualização das regiões de decisão com `matplotlib`.

---

### 🧪 DummyClassifier
- **Classe:** `DummyClassifier(strategy='stratified')`
- **Uso:** Comparar a performance dos modelos reais com um modelo de base aleatório.

---

## 📁 Experimentos e Dados

| Experimento               | Dados Usados                                                                                          | Modelo(s)         | Avaliação         | Destaques Visuais                  |
|---------------------------|--------------------------------------------------------------------------------------------------------|-------------------|-------------------|-------------------------------------|
| Venda de Carros           | [`precos.csv`](https://gist.githubusercontent.com/guilhermesilveira/dd7ba8142321c2c8aaa0ddd6c8862fcc/raw) | DecisionTree, SVM | `accuracy_score`  | Exportação da árvore (`graphviz`)  |
| Classificação de Animais  | Lista fixa no código                                                                                   | LinearSVC         | `accuracy_score`  | Simples e didático                 |
| Funil de Vendas           | [`tracking.csv`](https://gist.githubusercontent.com/guilhermesilveira/b9dd8e4b62b9e22ebcb9c8e89c271de4/raw) | SVM, Dummy        | `accuracy_score`  | Comparação real x aleatório       |
| Finalização de Projetos   | [`projetos.csv`](https://gist.githubusercontent.com/guilhermesilveira/12291c548acaf544596795709020e3db/raw) | LinearSVC, SVC     | `accuracy_score`  | Fronteiras de decisão com `plt`   |

---

## 📈 Visualizações

Inclui:
- Gráficos com `seaborn` para análise exploratória.
- Fronteiras de decisão com `matplotlib.contourf`.
- Exportação de árvores com `graphviz` para PDF.

---

## 🧪 Avaliação dos Modelos

Todos os modelos utilizam:
- **`accuracy_score`** como métrica principal.
- Divisão de dados com `train_test_split`.
- **Estratificação (`stratify`)** para manter a proporção das classes.

---

## 📚 Bibliotecas Utilizadas

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
  - `DecisionTreeClassifier`
  - `LinearSVC`
  - `SVC`
  - `DummyClassifier`
  - `train_test_split`
  - `StandardScaler`
  - `accuracy_score`
- `graphviz`

---

## 🛠️ Como Executar

1. Clone o repositório

2. pip install -r requirements.txt

3.python nome_do_script.py

---

✅ Objetivo

Este projeto foi desenvolvido para prática e estudo de machine learning com Python, testando classificadores, comparações com modelos de base e diferentes abordagens de visualização.


---

📎 Créditos

Conjuntos de dados por Guilherme Silveira, instrutor da Alura.

