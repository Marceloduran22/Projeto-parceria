import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


data_teste = pd.read_csv('./Dados/conjunto_de_teste.csv')
data = pd.read_csv('./Dados/conjunto_de_treinamento.csv')
testId = data_teste["Id"]

###########################################################################
## Arrumando os dados de ajuste do classificador
###########################################################################
## Removendo Variaveis
###########################################################################


x = data.drop(columns = 
              ['preco',
               'diferenciais',
               'Id',
               'tipo_vendedor',
               'churrasqueira',
               'quadra',
               's_jogos',
               's_ginastica',
               'area_extra',
               'tipo_vendedor'], axis=1)

y = data['preco']

###########################################################################
### Aplicandoo One Hot Encoding
###########################################################################

x = pd.get_dummies(x, columns = ['tipo'])
x = x.drop(columns = ['tipo_Loft', 'tipo_Quitinete'])
print(x.T)

###########################################################################
### Aplicando binarização
###########################################################################

binarizador = LabelEncoder()
x['bairro'] = binarizador.fit_transform(x['bairro'])

print( x.T)

###########################################################################
## Arrumando colunas com valores vazios
###########################################################################

imputer = SimpleImputer(strategy='most_frequent')
x = imputer.fit_transform(x)

###########################################################################
## Arrumando a escala dos valores
###########################################################################
StdSc = StandardScaler()
StdSc = StdSc.fit(x)
x = StdSc.transform(x)

###########################################################################
## Repetindo o mesmo processamento de dados para os dados de teste
###########################################################################
#data_teste = data_teste.drop([18548, 4406], axis = 0)

x_teste = data_teste.drop(columns = 
              ['diferenciais',
               'Id',
               'tipo_vendedor',
               'churrasqueira',
               'quadra',
               's_jogos',
               's_ginastica',
               'area_extra',
               'tipo_vendedor'], axis=1)

x_teste = pd.get_dummies(x_teste, columns = ['tipo'])
x_teste = x_teste.drop(columns = ['tipo_Loft'])

binarizador = LabelEncoder()
x_teste['bairro'] = binarizador.fit_transform(x_teste['bairro'])
    
imputer = SimpleImputer(strategy='most_frequent')
x_teste = imputer.fit_transform(x_teste)

StdSc = StandardScaler()
StdSc = StdSc.fit(x_teste)
x_teste = StdSc.transform(x_teste)


###########################################################################
## Treinando o classificador 
###########################################################################

def rmspe(y, y_resposta):
    value = np.sqrt(np.mean(np.square(((y - y_resposta) / y)), axis=0))
    return value

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regressorGB = HistGradientBoostingRegressor(l2_regularization=34, max_iter=140, loss = "absolute_error", max_depth=12)
regressorGB = regressorGB.fit(x_train, y_train)
gbPredictions = regressorGB.predict(x_test)
gbError = mean_squared_error(y_test, gbPredictions)
gbScore = r2_score(y_test, gbPredictions)
print("GB Error was: ", gbError)
print("GB Score was: ", gbScore)
print("GB rnmspe score wass: ", rmspe(y_test, gbPredictions))

regressorKNN = KNeighborsRegressor(n_neighbors=10, p=1, n_jobs=2, algorithm='kd_tree', weights='distance')
regressorKNN = regressorKNN.fit(x_train,y_train)
knnPredictions = regressorKNN.predict(x_test)
knnError = mean_squared_error(y_test, knnPredictions)
knnScore = r2_score(y_test, knnPredictions)
print("KNN Error was: ", knnError)
print("KNN Score was: ", knnScore)
print("KNN rnmspe score wass: ", rmspe(y_test, knnPredictions))

###########################################################################
## Prevendo os resultados e colocando eles no arquivo de resultados
###########################################################################

resposta = regressorGB.predict(x_teste)
prediction_file = {'Id': testId.index , 'preco': resposta}
prediction_file = pd.DataFrame(data=prediction_file)

prediction_file.to_csv('./Dados/Resultado.csv', index=False)
