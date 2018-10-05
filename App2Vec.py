from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import pickle
import collections


class App2Vec:
	def __init__(self,stop_app_path = []):
		'''
		training_data：Store the training data
		stop_app：Store the stop app. These stop apps won't treat as the training data for App2Vec.
		label2id：Store the mapping with cluster labels and app sequences.
		'''
		self.training_data = []
		self.stop_app = []
		self.label2id = collections.defaultdict(list)

		if stop_app_path:
			with open(stop_app_path,'r') as f:
				self.stop_app = f.read().split('\n')

	# half ignore process mode..
	def ignore_all_get_app(self,each_app_seq):
		each_app_list = each_app_seq.split()
		result = []
		for app in each_app_list:
			if app in self.stop_app:
				return []
			else:
				result.append(app)
		return [result]

	# provide the training data for App2Vec.
	def csv2training_data(self,raw_file_path,ignore_all = True):
		'''
		file_path：The storage location of your raw training data.
		ignore_all(Optional)：Ignore mode，True is Full ignore mode，False is half ignore mode.
		'''

		df = pd.read_csv(raw_file_path,header = None)

		for index,each_app_seq in df.iterrows():

			#Full ignore mode
			if ignore_all:
				for each_app_list in (map(self.ignore_all_get_app, each_app_seq.tolist())):
					self.training_data.extend(each_app_list)
			
			#Half ignore mode
			else:
				self.training_data.append([app for ele_app_list in each_app_seq.tolist() for app in each_app_list.split(' ') if app not in stop_app])
		
			
	#Train the app2vec.
	def training_App2Vec(self,app2vec_model_path):
		'''
		app2vec_model_path：The storage location of the app2vec model.
		'''

		#Views more, https://radimrehurek.com/gensim/models/word2vec.html
		model = Word2Vec(self.training_data,sg=1,size = 64,window = 3,seed = 0,min_count = 0,iter = 10,compute_loss=True)

		#save the model
		model.save(app2vec_model_path)