#!/usr/bin/env python
# coding: utf-8


import numpy as np

import keras
from keras import layers
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

from progressbar import ProgressBar, FormatLabel, Bar, Percentage

from timeit import default_timer as timer

import random
import glob
import os

import pandas as pd
import pickle
import Utils
from Utils import DataGenerator, TimingCallback, createModel

class NNManager():
	def __init__(self,  x_real = None,label_real = None,x_data = None,label_data = None,managerName = 'NNManager',
				 nNetworksFingerprint = 3,nEpochs = 15, networkSize = 100,accesstime = 480):
		if label_real is None:
			print('Buscando un NNManager para cargar...')
			
		else:
			self.instance = None
			'Initialization'
			self.managerName = managerName
			
			self.x_real = x_real
			self.label_real = label_real
			self.x_data = x_data
			self.label_data = label_data


			
			self.nNetworksFingerprint = nNetworksFingerprint
			self.nFingerprints = len(label_real)
			self.networkSize = networkSize
			self.nNetworks = (int(self.nFingerprints/self.networkSize)*self.nNetworksFingerprint) + 1
			self.currentNet = 0
			self.nEpochs = nEpochs
			self.dictionary = {}
			self.distribution = []
			self.models = []
			self.loadedModels = []
			
			self.labelDictionary = self.createLabelDictionary()
			
			self.printInfo()
			self.distributeNetworks()
			self.saveNNManager()

	
	def getInstance(self, managerName = 'Trained_NNManager'):

		instance = self.loadNNManager(self,managerName)
		return instance
	
	def saveNNManager(self):
		if not os.path.exists(self.managerName):
			os.makedirs(self.managerName)
			os.makedirs(self.managerName + '/saved_NNManager')
			os.makedirs(self.managerName + '/saved_models')
		
		self.models = []
		self.loadedModels = []
		binary_file = open(self.managerName + '/saved_NNManager/NNManData.bin','wb')
		d = pickle.dump(self, binary_file)
		binary_file.close()
		
	def loadNNManager( self, name = 'NNManager'):
		
		binary_file = open(name + '/saved_NNManager/NNManData.bin','rb')
		instance = pickle.load( binary_file )
		binary_file.close()
		
		instance.printInfo()
		instance.loadedModels = []
		
		return instance
	
	def printInfo(self):
		print ('NNManager: ', self.managerName)
		print ('Tenemos huellas de ',self.nFingerprints, ' individuos')
		print ('Tenemos ',self.nNetworks, ' redes entrenadas con sets de ',self.networkSize, ' huellas cada una')
		print ('Cada huella se encuentra en ',self.nNetworksFingerprint, ' redes como mínimo')
	
	def createLabelDictionary(self):
		label_real_dict = {}
		for i, y in enumerate(self.label_real):
			label = str(y)
			label_real_dict.setdefault(label,[]).append(i)
			
			#label_real_dict[key] = i
		self.labelDictionary = label_real_dict
		return self.labelDictionary
	
	def setDistribution(self,new_d):

		if len(new_d) == self.nNetworks and len(new_d[0]) == self.networkSize:
			self.distribution = new_d
		else:
			print('Muchacho, esta distribución no me sirve')
	
	def distributeNetworks(self):
		self.distribution = []
		step = 0#self.nNetworksFingerprint%self.nFingerprints
		usr = 0
		
		print('Iniciando distribuidor.')
		for netId in range(0,int(self.nNetworks)):
			d = []
			for netPosition in range (0,self.networkSize):
				label = str(self.label_real[(usr)])
				d.append((usr))
				usr = (usr + 1)%self.nFingerprints
				#print (usr)
			self.distribution.append(d)
		
		
		
		
		intercambios = max(self.nNetworks,self.networkSize)* self.nNetworks * self.nNetworks

		wdg = [FormatLabel('Realizando proceso de agitado. Número de sacudidas: %(value)d (tiempo: %(elapsed)s)'),
			  Percentage(), Bar()]
		pbar = ProgressBar( maxval=intercambios,widgets = wdg).start()
		
		colisiones  = 0
		for position in range(0,intercambios):
			pos2 = random.randint(0, self.networkSize)
			aux = self.distribution[position%self.nNetworks][pos2%self.networkSize]
			aux2 = self.distribution[(position +1)%self.nNetworks][(pos2)%self.networkSize]
			pbar.update(position+1)
			
			#Controlamos que no se repitan individuos dentro de la misma red
			if aux in self.distribution[(position +1)%self.nNetworks] or aux2 in self.distribution[(position)%self.nNetworks]:
				colisiones = colisiones +1
			else:
				self.distribution[position%self.nNetworks][pos2%self.networkSize] = aux2
				self.distribution[(position +1)%self.nNetworks][(pos2)%self.networkSize] = aux
		pbar.finish()
		
		#Creamos el diccionario
		self.dictionary = {}
		for netId in range(0,self.nNetworks):
			for pos2 in range (0,self.networkSize):
				usr = self.distribution[netId][pos2]
				label = str(self.label_real[(usr)])
				self.dictionary.setdefault(label,[]).append(netId)
		return self.distribution

	
	def trainNetworks(self):
		if len(self.distribution) == 0:
			print('Muchacho, esto no está distribuido')
		else:
			nw = 0
			for network in self.distribution:
				#if nw == 1: break
				modelName = 'Model_for_dataset_N_' + str(nw)
				
				x_r,label_r,x_d,label_d = createDataset_list_data(network,self.x_real,self.label_real,
																	  self.x_data,self.label_data)
	 
				x_data = np.concatenate([x_d], axis=0)
				label_data = np.concatenate([label_d], axis=0)
				
				#Partimos los datos entre Entrenamiento y Test
				x_train, x_val, label_train, label_val = train_test_split(x_data, label_data, test_size=0.1)
				
				#Creamos el diccionario para saber la posición de cada label
				label_real_dict = {}

				for i, y in enumerate(label_r):
					label = str(y)
					label_real_dict[label] = i
					#label_real_dict.setdefault(label,[]).append(i)
				
				#Generamos los datos de entrenamiento
				train_gen = DataGenerator(x_train, label_train, x_r, label_real_dict, shuffle=True)
				val_gen = DataGenerator(x_val, label_val, x_r, label_real_dict, shuffle=False)

				#Creamos el modelo
				md = createModel(modelName)
				

				cb = TimingCallback()
				cb.logs = []
				history = md.fit(train_gen, epochs=self.nEpochs, validation_data=val_gen)#,callbacks=[cb])
				
				#Guardamos el modelo entrenado
				md.save(self.managerName + '/saved_models/'+md.name+'.h5')
				nw = nw + 1
				
	def trainNetworks_user(self,userId):
		md = self.returnNetworks(userId)
		if md:
			for m in md:
				network = self.distribution[m]
				modelName = 'Model_for_dataset_N_' + str(m)
				print(modelName, ' se va a entrenar')
				
				x_r,label_r,x_d,label_d = createDataset_list_data(network,self.x_real,self.label_real,
																	  self.x_data,self.label_data)
	 
				x_data = np.concatenate([x_d], axis=0)
				label_data = np.concatenate([label_d], axis=0)
				
				#Partimos los datos entre Entrenamiento y Test
				x_train, x_val, label_train, label_val = train_test_split(x_data, label_data, test_size=0.1)
				
				
				
				#Creamos el diccionario para saber la posición de cada label
				label_real_dict = {}

				for i, y in enumerate(label_r):
					label = str(y)
					label_real_dict[label] = i
				
				#Generamos los datos de entrenamiento
				train_gen = DataGenerator(x_train, label_train, x_r, label_real_dict, shuffle=True)
				val_gen = DataGenerator(x_val, label_val, x_r, label_real_dict, shuffle=False)

				#Creamos el modelo
				md = createModel(modelName)
				
				#self.models.append(md)
				cb = TimingCallback()
				cb.logs = []
				history = md.fit(train_gen, epochs=self.nEpochs, validation_data=val_gen)#,callbacks=[cb])
				#Guardamos el modelo entrenado
				md.save(self.managerName + '/saved_models/'+md.name+'.h5')
		else:
			print('El individuo no existe')

	def saveNetworks(self):
		for md in self.models:
			md.save(self.managerName + '/saved_models/'+md.name+'.h5')

	def loadNetworks(self):
		mdls = sorted(glob.glob(self.managerName + '/saved_models/*.h5'))
		self.models = []
		self.loadedModels = []
		for file in mdls:
			new_model = keras.models.load_model(file)
			self.models.append(new_model)
			#TO DO: añadir a loadedModels

	def returnNetworks(self,userId):
		return self.dictionary.get(userId)
	
	def loadNetworks(self, userId):
		models_user = []
		md = self.returnNetworks(userId)
		if md:
			for m in md:
				if(m in self.loadedModels):
					print('Modelo entrenado con el dataset: ',m, 'ya cargado')
					loaded_model = self.models[self.loadedModels.index(m)]
					models_user.append(loaded_model)
				else:					
					file = glob.glob(self.managerName + '/saved_models/Model_for_dataset_N_'+ str(m) + '*.h5')
					if len(file) == 0:
						print('Red neuronal no encontrada. Precisa entrenamiento')
					else:
						print('Cargando modelo entrenado con el dataset: ',m)
						new_model = keras.models.load_model(file[0])
						self.models.append(new_model)
						self.loadedModels.append(m)
						models_user.append(new_model)
		else:
			print('El individuo no existe')
		return models_user
			
	def returnFingerprint(self,userId):
		return self.x_real[self.labelDictionary[userId]]
	
	
	def matchFingerprint(self,userId, fingerprint):
		pred_rx = []
		
		#Recuperamos la huella real del usuario con la que comparar
		try:
			rx = self.x_real[self.labelDictionary[userId]].reshape((1, 90, 90, 1)).astype(np.float32) / 255.
		except(KeyError):
			print('El usuario ', userId,' no existe')
			return pred_rx
		
		
		models_user = self.loadNetworks(userId)
		# Corremos el modelo sobre esta huella
		for m in models_user:
			pred_rx.append(m.predict([fingerprint, rx]))

		#self.models = []
		return pred_rx
	
	def distributeNewUser(self,userId):
		netId = self.nNetworks
		new_usr = self.nFingerprints

		for i in range(0,self.nNetworksFingerprint):
			d = []
			#Añadimos al nuevo usuario a la red y la incluimos en su diccionario
			d.append(self.nFingerprints)
			label = str(self.label_real[(new_usr)])
			self.dictionary.setdefault(label,[]).append(netId+i)
			
			for netPosition in range (1,self.networkSize):
				#Sacamos un usuario aleatorio de los que tenemos almacenados
				usr = (random.randint(0, self.nFingerprints - 1))%self.nFingerprints
				#Comprobamos que no lo hayamos añadido ya a la red
				while usr in d:
					#print(usr, ' ya estaba en la lista')
					usr = (random.randint(0, self.nFingerprints - 1))%self.nFingerprints
				#Obtenemos su ID y lo añadimos en el diccionario
				label = str(self.label_real[(usr)])
				self.dictionary.setdefault(label,[]).append(netId+i)
				#Añadimos el usuario aleatorio a la nueva red
				d.append((usr))
			self.distribution.append(d)
			self.nNetworks = self.nNetworks + 1

	
	def addUser(self,x_r_user , userId, x_data_user ,y_data_user, train = False):
		
		#añadimos los datos sobre el usuario a los datos que tenemos
		self.label_real = np.vstack((self.label_real,userId))
		#np.vstack((self.x_real,x_r_user))
		#print(np.vstack((self.x_real,x_r_user)))
		self.x_real = np.concatenate((self.x_real,x_r_user),axis = 0)
		
		self.x_data = np.concatenate((self.x_data,x_data_user),axis = 0)
		self.label_data = np.concatenate((self.label_data,y_data_user),axis = 0)

		label = str(userId)
		self.labelDictionary.setdefault(label,[]).append(self.nFingerprints)
		
		#realizamos la distribución y el diccionario para las n nuevas redes.
		self.distributeNewUser(str(userId))
		self.nFingerprints = self.nFingerprints +1	   
		
		#Guardamos el manager
		self.saveNNManager()
		
		#entrenamos las nuevas redes
		if(train):
			self.trainNetworks_user(str(userId))
			
	def deleteUser(self, userId, train = False):
		
		#buscamos al usuario en el diccionario
		try:
			#obtenemos su posición
			user_i = self.labelDictionary[str(userId)]
			user_img= self.x_real[self.labelDictionary[str(userId)]]
			#print(user_i[0], ' es el user_i')
			
			#Lo sacamos del diccionario de distribución y guardamos sus redes
			retorno = self.dictionary.pop(str(userId))
			#Para cada una de sus redes borramos el fichero si estaban entrenadas
			for r in retorno:
				modelName = 'Model_for_dataset_N_' + str(r)
				if os.path.exists(self.managerName + '/saved_models/'+modelName+'.h5'):
				  os.remove(self.managerName + '/saved_models/'+modelName+'.h5')
				  print('La red ',r,' ha sido borrada') 
				else:
				  print('La red ',r,' no estaba entrenada') 
			
				#Metemos un usuario aleatorio en cada una de las redes. Se modifica en la distribución y se añaden al diccionario
				usr = (random.randint(0, self.nFingerprints - 1))%self.nFingerprints-1
				#Comprobamos que no lo hayamos añadido ya a la red
				while usr in self.distribution[r]:
					usr = (random.randint(0, self.nFingerprints - 1))%self.nFingerprints-1
				#Obtenemos su ID y lo añadimos en el diccionario
				print('Añadido el usuario ',usr, '  la red ', r)
				label = str(self.label_real[(usr)])
				self.dictionary.setdefault(label,[]).append(r)
				
				#cambiamos el usuario a borrar por el usuario aleatorio en la red
				self.distribution[r] = [usr if x == user_i[0] else x for x in self.distribution[r]]

			#tomamos el último elemento añadido
			last = self.label_real[-1]
			last_i = self.labelDictionary[str(last)]
			
			#sacamos del diccionario de etiquetas al usuario a borrar
			self.labelDictionary.pop(str(userId))
			
			#sustituimos sus datos por los del último usuario
			self.x_real[user_i] = self.x_real[last_i]
			self.label_real[user_i] = self.label_real[last_i]
			
			
			#modificamos el diccionario de etiquetas de este usuario
			self.labelDictionary.pop(str(last))
			self.labelDictionary.setdefault(str(last),[]).append(user_i[0])
			last_i_post = self.labelDictionary[str(last)]
			
			#redimensionamos los datos label_real y x_real
			self.x_real = self.x_real[:-1]
			self.label_real = self.label_real[:-1]
			self.nFingerprints = self.nFingerprints - 1
	
			#Guardamos el manager
			self.saveNNManager()
				
		except(KeyError):
			print('El usuario ', userId,' no existe')


		

class Predictor():
	def __init__(self,managerName = 'Trained_NNManager'):
		try:
			
			self.manager = NNManager.getInstance(NNManager, managerName)
			print('Predictor Iniciado sobre el Gestor')
		except(FileNotFoundError):
			print('No existe un gestor llamado: ' ,managerName, '\nNo se ha creado un predictor')
			
			
	def predict(self,userId, fingerprint):
		return self.manager.matchFingerprint(userId, fingerprint)