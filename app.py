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
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3 ]
classes = [1, 1, 1, 0, 0, 0]

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(dados, classes)

animal_misterioso = [0, 0, 0]

predict = model.predict([animal_misterioso])

print(predict)
if predict == 0:
    print("Cachorro")
elif predict == 1:
    print("Porco")
