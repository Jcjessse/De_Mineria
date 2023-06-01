import pandas as pd #importamos pandas
import sklearn #biblioteca de aprendizaje automático
import matplotlib.pyplot as plt #Librería especializada en la creación de gráficos
from sklearn.decomposition import PCA #importamos algorimo PCA
from sklearn.decomposition import IncrementalPCA #importamos algorimo PCA
from sklearn.linear_model import LogisticRegression #clasificación y análisis predictivo
from sklearn.preprocessing import StandardScaler #Normalizar los datos
from sklearn.model_selection import train_test_split #permite hacer una división de un conjunto de datos en dos
#bloques de entrenamiento y prueba de un modelo
if __name__ == '__main__':
    dt_heart=pd.read_csv('./datos/Dunido.csv')
    print(dt_heart.head(5))
dt_features=dt_heart.drop(['INCIDENCIA'],axis=1) #las featurus sin el target
dt_target = dt_heart['INCIDENCIA'] #obtenemos el target
dt_features = StandardScaler().fit_transform(dt_features) #Normalizamnos los datos
X_train,X_test,y_train,y_test =train_test_split(dt_features,dt_target,test_size=0.30,random_state=42)
print(X_train.shape) #consultar la forma de la tabla con pandas
print(y_train.shape)
#EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará
#de esta forma:
#a: n_components = min(n_muestras, n_features)
pca=PCA(n_components=3)
# Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
pca.fit(X_train)
#Como haremos una comparación con incremental PCA, haremos lo mismo para el IPCA.
PCA
#EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
#poco a poco y combinarlos en el resultado final
ipca=IncrementalPCA(n_components=3,batch_size=10) #tamaño de bloques, no manda a entrear todos los datos
#Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
ipca.fit(X_train)
#Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
#me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
#en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
#para nuestro modelo
plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_) #gneera desde 0 hasta los componentes
plt.show()
#Ahora vamos a configurar nuestra regresión logística
logistic=LogisticRegression(solver='lbfgs')
# Configuramos los datos de entrenamiento
dt_train = pca.transform(X_train)#conjunto de entrenamiento
dt_test = pca.transform(X_test)#conjunto de prueba
# Mandamos los data frames la la regresión logística
logistic.fit(dt_train, y_train) #mandasmos a regresion logistica los dos datasets
#Calculamos nuestra exactitud de nuestra predicción
print("SCORE PCA: ", logistic.score(dt_test, y_test))
#Configuramos los datos de entrenamiento
dt_train = ipca.transform(X_train)
dt_test = ipca.transform(X_test)
# Mandamos los data frames la la regresión logística
logistic.fit(dt_train, y_train)
#Calculamos nuestra exactitud de nuestra predicción
print("SCORE IPCA: ", logistic.score(dt_test, y_test))