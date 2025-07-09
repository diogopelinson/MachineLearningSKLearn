# Features [1 sim, 0 nao]
# pelo longo?
# perna curta?
# faz auau?

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 -  Porco,  0 - Cachorro
# Dados = x , classes = y
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3 ]
treino_y = [1, 1, 1, 0, 0, 0]

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(treino_x, treino_y)

animal_misterioso = [0, 0, 0]

predict = model.predict([animal_misterioso])

print(predict)
if predict == 0:
    print("Cachorro")
elif predict == 1:
    print("Porco")

print("---------------------")

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

teste_x = [misterio1, misterio2, misterio3]
previsoes = model.predict(teste_x)

teste_y = [0,1,1]

print(previsoes)

correto = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_acerto = (correto / total) * 100

# Acurácia: proximidade entre o valor obtido experimentalmente e o valor verdadeiro na medição de uma grandeza física.
print(f"Acurácia: {taxa_acerto:.2f}%")


from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(teste_y, previsoes) * 100
print(f"Acurácia: {taxa_acerto:.2f}%")


print("---------------------")