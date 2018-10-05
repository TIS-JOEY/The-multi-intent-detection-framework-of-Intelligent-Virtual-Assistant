import numpy as np
from sklearn.externals import joblib
import pickle
import collections
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation

class Affinity_Propagation:
	def __init__(self,app2vec_model_path,af_model_path,prefer,training_data):
		self.app2vec_model_path = app2vec_model_path
		self.af_model_path = af_model_path
		self.prefer = prefer
		self.training_data = training_data
		self.label2id = collections.defaultdict(list)

	def train(self):
		
		#load app2vec model.
		app2vec_model = Word2Vec.load(self.app2vec_model_path)

		#get the vector of app2vec.
		vector = app2vec_model.wv.syn0

		#store the training data of AF.
		af_training_data = []

		#average the vector of each app sequence as a unit
		for app_seq in self.training_data:
			af_training_data.append(np.mean([app2vec_model[app] for app in app_seq],0))

		# train the af model.
		af_model = AffinityPropagation(preference = self.prefer).fit(af_training_data)

		# save the model
		joblib.dump(af_model, self.af_model_path)


	def get_label2id(self):
		# load af model
		af = joblib.load(self.af_model_path)

		# build a label2id dictionary
		for index,label in enumerate(af.labels_):
			self.label2id[label].append(index)