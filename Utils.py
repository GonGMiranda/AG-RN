#!/usr/bin/env python
# coding: utf-8


import numpy as np

import keras
from keras import layers
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


from timeit import default_timer as timer

import random
import glob
import os

import pandas as pd
import pickle

def createDataset_X(x_real,y_real,x_easy,y_easy,x_medium, y_medium, x_hard, y_hard, dataSetSize): 
	random_users = []
	for i in range(0,dataSetSize):
		random_users.append(random.randint(0, len(x_real)-1))
	random_users.sort()	
	#print(random_users)
		
	y_real_X = []
	x_real_X = []
	y_easy_X = []
	x_easy_X = []
	y_medium_X = []
	x_medium_X = []
	y_hard_X = []
	x_hard_X = []
	
	
	for usr in random_users:
		label_user = y_real[usr]
		y_real_X, x_real_X = addUser(y_real,x_real,label_user,y_real_X, x_real_X)
		y_easy_X, x_easy_X = addUser(y_easy,x_easy,label_user,y_easy_X, x_easy_X)
		y_medium_X, x_medium_X = addUser(y_medium,x_medium,label_user,y_medium_X, x_medium_X)
		y_hard_X, x_hard_X = addUser(y_hard,x_hard,label_user,y_hard_X, x_hard_X)

	return x_real_X,y_real_X,x_easy_X,y_easy_X,x_medium_X, y_medium_X, x_hard_X, y_hard_X

def createDataset_list_data(usrs_list,x_real,y_real,x_data,y_data):
		
	y_real_X = []
	x_real_X = []
	y_data_X = []
	x_data_X = []
	
	idx_full = []

	for usr in usrs_list:

		label_user = y_real[usr]

		y_real_X.append(label_user)
		x_real_X.append(x_real[usr])
		
		indexesData = [i for i,label in enumerate(y_data) if label[0] == label_user[0] and 
					  label[1] == label_user[1] and label[2] == label_user[2] and label[3] == label_user[3]]
		for i in indexesData:
			idx_full.append(i)
	
	for idx in idx_full:
		y_data_X.append(y_data[idx])
		x_data_X.append(x_data[idx])


	return x_real_X,y_real_X,x_data_X,y_data_X

def addUser(labels,imgs, label_user, labels_Usr = [],imgs_Usr = []):
	for i in range(0,len(labels)):
		if labels[i][0] == label_user[0] and labels[i][1] == label_user[1] and labels[i][2] == label_user[2] and labels[i][3] == label_user[3]:
			labels_Usr.append(labels[i])
			imgs_Usr.append(imgs[i])
	return labels_Usr,imgs_Usr


def dataAugmentationImg(img):
	seq = iaa.Sequential([
			iaa.GaussianBlur(sigma=(0, 0.5)),
			iaa.Affine(
				scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
				translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
				rotate=(-30, 30),
				order=[0, 1],
				cval=255
			)
		], random_order=True)

	return  seq.augment_image(img).reshape((1, 90, 90, 1)).astype(np.float32) / 255.
	
def dataAugmentationImgs(imgs):
	seq = iaa.Sequential([
			iaa.GaussianBlur(sigma=(0, 0.5)),
			iaa.Affine(
				scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
				translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
				rotate=(-30, 30),
				order=[0, 1],
				cval=255
			)
		], random_order=True)

	return  seq.augment_images(imgs)



class DataGenerator(keras.utils.Sequence):
	def __init__(self, x, label, x_real, labelDictionary, batch_size=32, shuffle=True):	  
		'Initialization'
		self.x = x
		self.label = label
		self.x_real = x_real
		self.labelDictionary = labelDictionary
		
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.x) / self.batch_size))

	def __getitem__(self, index):
		'Genera un batch de datos'
		# Genera los índices del batch
		x1_batch = self.x[index*self.batch_size:(index+1)*self.batch_size]
		label_batch = self.label[index*self.batch_size:(index+1)*self.batch_size]
		
		x2_batch = np.empty((self.batch_size, 90, 90), dtype=np.float32)
		y_batch = np.zeros((self.batch_size, 1), dtype=np.float32)
		
		# Se realiza el proceso de Data augmentation
		x1_batch = dataAugmentationImgs(x1_batch)
		
		# se toman etiquetas que matchean (label 1.0) y que no matchean (label 0.0) y se añaden todas en el mismo batch
		# todas las imágenes que matcheen tienen que ser de la misma huella.
		for i, l in enumerate(label_batch):
			match_key = str(l)

			if random.random() > 0.5:
				# imagen que matchea
				x2_batch[i] = self.x_real[self.labelDictionary[match_key]]
				y_batch[i] = 1.
			else:
				# imagen que no matchea
				while True:
					unmatch_key, unmatch_idx = random.choice(list(self.labelDictionary.items()))

					if unmatch_key != match_key:
						break

				x2_batch[i] = self.x_real[unmatch_idx]
				y_batch[i] = 0.

		return [x1_batch.astype(np.float32) / 255., x2_batch.astype(np.float32) / 255.], y_batch

	def on_epoch_end(self):
		if self.shuffle == True:
			self.x, self.label = shuffle(self.x, self.label)
			
		

class TimingCallback(keras.callbacks.Callback):
	def __init__(self, logs={}):
		self.logs=[]
	def on_epoch_begin(self, epoch, logs={}):
		self.starttime = timer()
	def on_epoch_end(self, epoch, logs={}):
		self.logs.append(timer()-self.starttime)
		
def createModel(modelName):
	fp1 = layers.Input(shape=(90, 90, 1))
	fp2 = layers.Input(shape=(90, 90, 1))

	# ambas capas comparten pesos
	inputs = layers.Input(shape=(90, 90, 1))

	feat = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
	feat = layers.MaxPooling2D(pool_size=2)(feat)

	feat = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(feat)
	feat = layers.MaxPooling2D(pool_size=2)(feat)

	feature_model = Model(inputs=inputs, outputs=feat, name  = 'featureModel')

	# modelos de features que comparten pesos
	fp1_net = feature_model(fp1)
	fp2_net = feature_model(fp2)

	# hacemos un subtract de las features
	net = layers.Subtract()([fp1_net, fp2_net])

	net = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(net)
	net = layers.MaxPooling2D(pool_size=2)(net)

	net = layers.Flatten()(net)
	net = layers.Dense(64, activation='relu')(net)
	net = layers.Dense(1, activation='sigmoid')(net)

	model = Model(inputs=[fp1, fp2], outputs=net, name = modelName)

	model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['acc'])
	model.summary()
	
	return model

def findDataUser(labels,imgs, label_user, labels_Usr = [],imgs_Usr = []):
	labels_Usr = []
	imgs_Usr = []
	for i in range(0,len(labels)):
		if labels[i][0] == label_user[0] and labels[i][1] == label_user[1] and labels[i][2] == label_user[2] and labels[i][3] == label_user[3]:
			labels_Usr.append(labels[i])
			imgs_Usr.append(imgs[i])
	return labels_Usr,imgs_Usr 


		
		