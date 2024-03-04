from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from perceptron import Perceptron as Perceptron

num_particiones = input("¿Cuantas particiones?: ")
num_porcentaje_entrenamiento = input("¿Que porcentaje de entrenamiento? Ej: 80 (El restante sera de prueba): ")

num_particiones = int(num_particiones)
num_porcentaje_entrenamiento = int(num_porcentaje_entrenamiento)

str_archivo_usado = 'spheres1d10.csv'

csvfile = open(str_archivo_usado, 'r').readlines()

num_len_particion = len(csvfile) / num_particiones
num_len_particion = int(num_len_particion)

filename = 1
nombres = []
for i in range(len(csvfile)):
	if i % num_len_particion == 0:
		str_temp = str_archivo_usado + str(filename) + '.csv'
		nombres.append(str_temp)
		open(str_temp, 'w+').writelines(csvfile[i:i+num_len_particion])
		filename += 1

print("Se generaron las siguentes " + str(num_particiones) + " particiones: ")
for i in nombres:
	print(i)
print("Con el " + str(num_porcentaje_entrenamiento) + "% para el entrenamiento, y el " + str(100-num_porcentaje_entrenamiento) + "% para la generalizacion")

filename = 1

n_iter = 50

for i in range(num_particiones):
	
	df = pd.read_csv(str_archivo_usado + str(filename) + '.csv', header=None)
	filename += 1
	num_datasetsize = len(df)
	num_entrenamiento = (num_porcentaje_entrenamiento/10) * num_datasetsize

	num_entrenamiento = int(num_entrenamiento)
	num_datasetsize = int(num_datasetsize)

#----------------------------------------------------------------------------------------------------------------------

	print("Cargando archivo de entrenamiento...")

	X1 = df.iloc[0:num_entrenamiento, [0, 1, 2]].values
	y1= df.iloc[0:num_entrenamiento, 3].values
	y1= np.where(y1== -1, -1, 1)

	ppn = Perceptron(aprendizaje=0.1, n_iter=n_iter)
	print("Entrenando...")
	ppn.fit(X1, y1) 

	print("Mostrando resultados...")
	fig = plt.figure(0)
	plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
	plt.xlabel("Iteracion")
	plt.ylabel("Numero de errores")

	plt.show()
#------------------------------------------------
	trn_file = np.array(df.iloc[0:num_entrenamiento, [0, 1, 2]])
	y = np.array(df.iloc[0:num_entrenamiento, 3])

	# Crear una figura tridimensional
	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')

	# Graficar los puntos en 3D
	ax.scatter(trn_file[:, 0], trn_file[:, 1], trn_file[:, 2], c=y)

	# Etiquetas de los ejes
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('x3')

	# Mostrar el gráfico
	plt.show()

#----------------------------------------------------------------------------------------------------------------------

	print("Cargando archivo de prueba...")

	X2 = df.iloc[num_entrenamiento:num_datasetsize, [0, 1, 2]].values
	y2= df.iloc[num_entrenamiento:num_datasetsize, 3].values
	y2= np.where(y2== -1, -1, 1)

	print("Probando...")
	ppn.test(X2, y2) 

	print("Mostrando resultados...")

	fig = plt.figure(2)
	plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
	plt.xlabel("Iteracion")
	plt.ylabel("Numero de errores")

	plt.show()
#------------------------------------------------
	tst_file = np.array(df.iloc[num_entrenamiento:num_datasetsize, [0, 1, 2]])
	y = np.array(df.iloc[num_entrenamiento:num_datasetsize, 3])

	# Crear una figura tridimensional
	fig = plt.figure(3)
	ax = fig.add_subplot(111, projection='3d')

	# Graficar los puntos en 3D
	ax.scatter(trn_file[:, 0], trn_file[:, 1], trn_file[:, 2], c=y)

	# Etiquetas de los ejes
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('x3')

	# Mostrar el gráfico
	plt.show()

#----------------------------------------------------------------------------------------------------------------------