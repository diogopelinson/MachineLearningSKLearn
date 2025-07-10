🧠 Machine Learning com Scikit-Learn — Classificadores e Visualização

Este repositório contém diversos experimentos de aprendizado de máquina utilizando a biblioteca Scikit-Learn. Os dados são públicos e foram extraídos de gists para projetos didáticos, com foco em classificação supervisionada, avaliação de modelos e visualizações de decisão.


---

📊 Modelos Utilizados

🌳 Decision Tree Classifier

Modelo: DecisionTreeClassifier(max_depth=3)

Aplicação: Previsão de venda de carros com base em preço, idade e km por ano.

Visualização: Exportação da árvore de decisão em PDF com graphviz.


🔍 LinearSVC

Modelo: LinearSVC()

Aplicações:

Classificação de animais com base em atributos físicos.

Previsão de conversão de leads (ex: palestra, contato, patrocínio).

Classificação de projetos finalizados com base no tempo estimado e preço.



📈 SVC (Support Vector Classifier)

Modelo: SVC(gamma='auto')

Aplicações:

Mesmos dados dos projetos.

Com e sem normalização dos dados.

Utilização de StandardScaler para melhora de performance.

Visualização das regiões de decisão com matplotlib.



🧪 DummyClassifier

Modelo: DummyClassifier(strategy='stratified')

Função: Comparar a performance dos modelos reais com um modelo base (aleatório).



---

📁 Estrutura dos Experimentos

Experimento	Dados Usados	Modelo	Avaliação	Destaques

Venda de carros	precos.csv	Árvores, SVM	accuracy_score	Conversão de milhas para km, idade do carro, visualização da árvore
Classificação animal	Lista estática	LinearSVC	accuracy_score	Identificação binária de "porcos" e "cachorros"
Funil de vendas	tracking.csv	SVM, Dummy	accuracy_score	Avaliação de conversão de leads
Projetos finalizados	projetos.csv	LinearSVC, SVC	accuracy_score, visual	Visualização de fronteiras de decisão



---

📈 Visualizações

As visualizações incluídas:

Gráficos de dispersão com seaborn

Fronteiras de decisão com contourf e matplotlib

Exportação de árvores de decisão com graphviz



---

🧪 Avaliação dos Modelos

Todos os modelos foram avaliados com métrica de acurácia, com separação treino/teste usando train_test_split e estratificação para manter a proporção das classes.


---

📚 Bibliotecas Utilizadas

pandas

numpy

sklearn.model_selection, sklearn.metrics, sklearn.tree, sklearn.svm, sklearn.dummy, sklearn.preprocessing

matplotlib.pyplot

seaborn

graphviz



---

🛠️ Como Executar

1. Clone este repositório:

git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo


2. Instale as dependências:

pip install -r requirements.txt


3. Execute os scripts Python desejados com:

python nome_do_script.py




---

✅ Objetivo

Este projeto foi criado para estudos e experimentos práticos de machine learning com Python, testando modelos lineares, não lineares, árvores de decisão e conceitos de baseline.


---

📎 Créditos

Conjuntos de dados por Guilherme Silveira, instrutor da Alura.


---