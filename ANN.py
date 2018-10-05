from annoy import AnnoyIndex
from gensim.models import Word2Vec


class ANN:
	def __init__(self):
		'''
		dim = the Dimension of App2Vec.
		num_tree：The number of trees of your ANN forest. More tress more accurate.
		ann_model_path：The storage path of ANN model.
		app2vec_mode_path：The storage path of app2vec model.
		'''

		self.dim = dim
		self.num_tree = num_tree
		self.app2vec_model_path = app2vec_model_path
		self.ann_model_path = ann_model_path

	#Train the ANN
	def train(self):

		#load app2vec model.
		app2vec_model = Word2Vec.load(self.app2vec_model_path)

		#get the vector of app2vec.
		vector = app2vec_model.wv.syn0
		
		t = AnnoyIndex(self.dim)
		
		for i in app2vec_model.wv.vocab:
			#get the mapping id.
			index = app2vec_model.wv.vocab[str(i)].index

			#add the mapping.
			t.add_item(index,vector[index])

		#train the app2vec. num_tree is the number of your ANN forest.
		t.build(self.num_tree)

		#save the model
		t.save(self.ann_model_path)