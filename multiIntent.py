import re
from aip import AipNlp
from watson_developer_cloud import ConversationV1
import os
import numpy as np
from multiprocessing import Process, Manager,Lock
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from collections import Counter
import Training.Model


class EMIP:
	'''
	Process explict multi-intent.
	'''
	def __init__(self,workspace_id):
		manager = Manager()
		self.baidu_nlp = None
		self.watson_nlp = None
		self.conjunction = ['和','還有','然後','或者','或','及','跟','與','或','以及','並且','並','而且','再來','因此','因為','所以','由於','不但','不僅','而且','以便']
		self.conj = conj
		self.score_saver = manager.dict()
		self.intent_saver = manager.dict()
		self.success_saver = manager.dict()
		self.failed_saver = manager.dict()
		self.lock = Lock()
		self.verb = []
		self.entities = []
		self.workspace_id = workspace_id


	def baidu_api(self,APP_ID,API_KEY,SECRET_KEY):
		'''
		Load Baidu NLP API.
		'''

		self.baidu_nlp = AipNlp(APP_ID,API_KEY,SECRET_KEY)

	def watson_api(self,usr_name,passwd):
		'''
		Load Watson NLP API.
		'''
		self.watson_nlp = ConversationV1(
		username = usr_name,
		password = passwd,
		version='2017-04-21')

	def detect_entities(self,text):
		"""
		Detects entities in the text by Google NLP API.
		"""

		client = language.LanguageServiceClient()

		if isinstance(text, six.binary_type):
			text = text.decode('utf-8')

		# Instantiates a plain text document.
		document = types.Document(
			content=text,
			type=enums.Document.Type.PLAIN_TEXT)

		# Detects entities in the document. You can also analyze HTML with:
		#   document.type == enums.Document.Type.HTML
		entities = client.analyze_entities(document).entities

		# entity types from enums.Entity.Type
		entity_type = ('UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION',
					   'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER')

		entity = list(set(i for i in entities))

		self.entities.extend(entity)

	def detectVerb(self,sen):
		"""
		Detect Verbs in text by Baidu NLP API.
		"""
		data = self.baidu_api.lexer(sen)['items']
		detect_verb = [i['item'] for i in data if i['pos']=='v']
		self.verb.extend(detectVerb)

	def detectIntent(self,sentence):
		'''
		Detect Intent in text by Watson NLP API.
		'''
		
		if os.getenv("conversation_workspace_id") is not None:
			self.workspace_id = os.getenv("conversation_workspace_id")

		response = self.watson_api.message(workspace_id = self.workspace_id, input={'text': sentence})

		if 'intents' in response:
			return response['entities'],response['intents'][0]['confidence'],response['intents'][0]['intent']
		else:
			return None,None,None

	def detectConj(self,sen):
		'''
		Detect conjunctions in the text and return the number of them.
		'''
		for i in self.conj:
			result = [m.start() for m in re.finditer(i,sen)]
			self.conjunction.extend(result)
		return len(result)

	def processConj(self,sen,local):
		'''
		Find the best separating point by statistic language model.
		'''
		resultConj = None

		# location conjunction processing.
		if local:
			minScore = float('inf')
			answer = []

		# global conjunction processing.
		else:
			minScore = self.baidu_nlp.dnnlm(sen)['ppl']
			answer = [sen]


		for conj in self.conj:
			if(conj not in sen):
				continue

			# find the location of conj in input sen.
			conj_pos = [m.start() for m in re.finditer(conj,sen)]
			for pos in conj_pos:

				# Base on the position of each conj, we separate the input sentence to leftSen and rightSen.
				leftSen = text[:pos]
				rightSen = text[pos+len(conj):]

				if(rightSen==''):
					continue

				# process leftSen...
				if(leftSen not in self.score_saver):
					leftScore = self.baidu_nlp.dnnlm(leftSen)['ppl']
					self.self.score_saver[leftSen] = leftScore
				else:
					leftScore = self.self.score_saver[leftSen]


				# process rightSen...
				if(rightSen not in self.self.score_saver):
					rightScore = self.baidu_nlp.dnnlm(rightSen)['ppl']
					self.self.score_saver[rightSen] = rightScore
				else:
					rightScore = self.self.score_saver[rightSen]

				# Treat the average score as the score of this conj.
				score = (leftScore+rightScore)/2

				# Find the conj with smallest perplexity score.
				if(minScore>score):
					resultConj = i
					answer = [leftSen,rightSen]

		return (answer,resultConj)

	def global_process(self,sen,conj = ''):
		if(conj == ''):
			conj_count = self.detectConj(sen = sen)
		else:
			conj_count = 1

		
		if conj_count:
			# Find the best separating point.
			ans,conj = self.processConj(sen, local = False)

			# The input sentence has the smallest perplexity.
			# This means there is no need to cut the input sentence.
			# So input it to local_process.
			if conj == None:
				self.local_process(sen = sen,global_sen = sen)

			# Separate the input sentence to two part and input each of them to process separately.
			elif ans:
				for i in ans:
					p = Process(target = self.global_process,args=(i,conj))
					p.start()
					p.join()

		# don't have any conj.
		else:
			self.local_process(sen = sen,global_sen = sen,conj_count = 0)

	def local_process(self,sen,global_sen,global_intent = '',conj = '',conj_count = None):
		if(conj_count == None):
			conj_count = self.detectConjunction(sen)
		
		
		if conj_count:
			# Find the best separating point.
			ans,conj = self.processConj(sen, local = True)

			# Separate the input sentence to two part and input each of them to process separately.
			if ans:
				for i in ans:
					p = Process(target = self.localProcess,args=(i,global_sen,intent))
					p.start()
					p.join()

			# There is no need to cut the input sentence.
			# Detect the intent and entities.
			else:
				intent,ent,score = self.detectIntent(sen)
				self.lock.acquire()
				self.intent_saver[sen] = intent
				self.success_saver[sen] = [sen,intent,global_sen]
				self.lock.release()

		# Don't have any conj.
		else:

			# Detect the intent and entities.
			intent,ent,score = self.detectIntent(sen)

			# Recognized
			if intent:
				self.lock.acquire()
				self.intent_saver[sen] = intent
				self.success_saver[sen] = [sen,intent,global_sen]
				self.lock.release()

			# Unrecognized
			else:
				self.lock.acquire()
				self.failed_saver[sen] = [sen,global_sen]
				self.saver[global_sen] = global_intent
				self.success_saver[global_sen] = [global_sen,global_intent,global_sen]
				self.lock.release()

	def contextProcess(self,global_sen):
		'''
		Guess the intent by their context.
		'''
		if(type(self.intent_saver)!=list):
			self.intent_saver = list(self.intent_saver.values())

		self.success_saver = list(self.success_saver.values())
		self.failed_saver = list(self.failed_saver.values())

		# There don't have any unrecognized record.
		if self.failed_saver:
			return None
		else:
			# Detect verbs in global_sen.
			self.detectVerb(global_sen)

			# Detect entities in global_sen.
			self.detect_entities(global_sen)

			# process unrecognized records.
			for failed_sen in self.failed_saver:
				judge = True

				# Guess the intent of failed_sen by their context.
				guessIntent = self.contextGuess(failed_sen[0],failed_sen[1],global_sen)

				# Recognized
				if guessIntent:
					for success_sen in self.success_saver:

						# Already recognize
						if(success_sen[2]==failed_sen[1] and guessIntent==failed_sen[1]):
							judge = False
							break

					# Recognized
					if judge:
						self.success_saver.append(guessIntent)
					else:
						continue

	def contextGuess(self,sen,candidate,global_sen):
		'''
		Add contexts and detect the intent again.
		'''
		guessIntent = None

		# collect the entities in this unrecognized record.
		entitiesOri = [i for i in self.entities if i in sen]

		if(entitiesOri==[]):

			# collect the entities in the relevant sub_sentence.
			entitiesTar = [i for i in self.entities if i in candidate]

			# Add the context(entities) and attempt to detect the intent.
			for i in entitiesTar:

				# Ignore enetities which already in this unrecognized record.
				if(i in sen):
					continue

				# Add the context.
				tmpS = sen+i

				minScore = -1
				intent,ent,score = self.detectIntent(tmpS)

				# Unrecognized
				if(score == None):
					continue

				# Find the intent with best recognized confidence score.
				if(minScore<score):
					guessIntent = intent

			# Recognized
			if guessIntent:
				return guessIntent
		

		posCan = global_sen.find(candidate)
		posEndCan = posCan+len(candidate)-1
		v = [i[1] for i in self.verb if i[1] in candidate and int(i[0])>=posCan and int(i[0])<=posEndCan]


		# Add the context(verbs) and attempt to detect the intent.
		for i in v:
			# Ignore verbs which already in this unrecognized record.
			if(i in sen):
				continue

			# Add the context.
			tmpS = i+sen

			minScore = -1
			intent,ent,score = self.detectIntent(tmpS)

			# Unrecognized
			if(score == None):
				continue

			# Find the intent with best recognized confidence score.
			if(minScore<score):
				guessIntent = intent

		# Recognized
		if guessIntent:
			return guessIntent

	def getIntent(self):
		return self.intent_saver

def calculateJob(sen,candidate,CanDes,shared_dict):
	score = calDocScore(sen,CanDes)
	shared_dict[candidate] = score

def calDocScore(text1,text2):
	APP_ID = "API_ID"
	API_KEY = "API_KEY"
	SECRET_KEY = "SECRET_KEY"
	client = AipNlp(APP_ID,API_KEY,SECRET_KEY)
	return client.simnet(text1,text2)['score']

class IMIP:
	def __init__(self,explicit_intent,intentApp,app2vec_model_path,ann_model_path,af_model_path):

		# The mapping between apps and intents
		self.intentApp = intentApp

		# Store the explicit intents
		self.explicit_intent = explicit_intent

		# Initial App2Vec class
		self.app2vec = Model.App2Vec()

		# Initial ANN class
		self.ann = Model.ANN(app2vec_model_path = app2vec_model_path,ann_model_path = ann_model_path)

		# Initial AF class
		self.af = Model.AF(app2vec_model_path = app2vec_model_path,af_model_path = af_model_path)

		# Initial BILSTM class
		self.bilstm = Model.BILSTM(app2vec_model_path = app2vec_model_path,max_len = 5)

		# Initial processData class
		self.p_data = Model.processData()

		# Set up description
		self.p_data.processDescription()

		# Load App2Vec model
		self.app2vec_model = app2vec.load_App2Vec(app2vec_model_path)

	def query(self,input_sen = None,model = 'ANN',doc = True,lstm = True):
		if model == 'ANN':
			if lstm:
				if doc:
					results = self.BILSTM_ANN_with_doc_process(input_sen)
				else:
					results = self.BILSTM_ANN_without_doc_process()
			else:
				if doc:
					results = self.ANN_with_doc_process(input_sen)
				else:
					results = self.ANN_without_doc_process()

		else:
			if lstm:
				if doc:
					results = self.BILSTM_AF_with_doc_process(input_sen)
				else:
					results = self.BILSTM_AF_without_doc_process()
			else:
				if doc:
					results = self.AF_with_doc_process(input_sen)
				else:
					results = self.AF_without_doc_process()

		return results


	def ANN_without_doc_process(self):

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get ids
		indexs = [self.app2vec_model.wv.vocab[app].index+1 for app in apps]

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([self.ann_model.get_nns_by_item(index,5) for index in indexs]))
		
		# Transfer to app and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr-1] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr-1] not in apps]

		counter = collections.Counter(nbrs)

		most_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

		result = self.p_data.checkClass(most_voting_filter,5)

		return result

	def ANN_with_doc_process(self,input_sen):

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# Get ids
		indexs = [self.app2vec_model.wv.vocab[app].index+1 for app in apps]

		# Get their neighbor and flat it to 1D.
		nbrs = list(itertools.chain.from_iterable([self.ann_model.get_nns_by_item(index,5) for index in indexs]))
		
		# Transfer to app and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr-1] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for nbr_id in range(len(nbr_app)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen, nbr_app[nbr_id],self.p_data.app2des[nbr_app[nbr_id]],shared_dict))

		pool.close()
		pool.join()
				
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

		return result

	def AF_without_doc_process(self):

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps],0)

		# The predicted label
		predict_label = af_model.predict([vector])

		# Major voting 
		counter = collections.Counter(self.af.label2app[predict_label[0]])

		# Choose the top k apps with higher voting and avoid duplicate.
		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common() if app_with_count[0] not in apps]

		result = self.p_data.checkClass(major_voting_filter,len(y_test[app_seq_id]))
		
	def AF_with_doc_process(self,input_sen):

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [intentApp[i] for i in self.explicit_intent]

		# Get the input vector
		vector = np.mean([self.app2vec_model[app] for app in apps],0)

		# The predicted label
		predict_label = af_model.predict([vector])

		# Get the candididate apps and avoid duplicate
		candidiates = list(filter(lambda x:x not in apps,list(set(self.af.label2app[predict_label[0]]))))

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for candidiate_id in range(len(candidiates)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen,candidiates[candidiate_id],p_data.app2des[candidiates[candidiate_id]]))

		pool.close()
		pool.join()

		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

		return result

	def BILSTM_ANN_without_doc_process(self,filepath = 'data/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to index
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# Get their neighbors.
		nbrs = ann_model.get_nns_by_vector(vector_predict,10)

		# Transfer them to apps and avoid duplicate
		nbrs = [self.app2vec_model.wv.index2word[nbr - 1] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		counter = collections.Counter(nbrs)

		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common()]

		result = self.p_data.checkClass(major_voting_filter,5)

	def BILSTM_ANN_with_doc_process(self,input_sen, filepath = 'data/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(filepath)

		# Load ANN model
		ann_model = self.ann.load_ANN()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# Get their neighbor and flat it to 1D.
		nbrs = ann_model.get_nns_by_vector(vector_predict,len(y_test[app_seq_id]))

		# Transfer to app
		nbr_app = [self.app2vec_model.wv.index2word[nbr - 1] for nbr in nbrs if self.app2vec_model.wv.index2word[nbr] not in apps]

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for nbr_id in range(len(nbr_app)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen, nbr_app[nbr_id],self.p_data.app2des[nbr_app[nbr_id]],shared_dict))

		pool.close()
		pool.join()
				
		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

	def BILSTM_AF_without_doc_process(self,filepath = 'data/BILSTM_model.h5'):

		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		# Major voting 
		counter = collections.Counter(self.af.label2app[predict_label[0]])

		# Choose the top k apps with higher voting and avoid duplicate.
		major_voting_filter = [app_with_count[0] for app_with_count in counter.most_common() if app_with_count[0] not in apps]

		result = self.p_data.checkClass(major_voting_filter,5)

	def BILSTM_AF_with_doc_process(self,input_sen,filepath = 'data/BILSTM_model.h5'):
		
		# Load BILSTM model
		bilstm_model = self.bilstm.load_BI_LSTM_model(filepath)

		# Load AF model
		af_model = self.af.get_af_model()

		# transfer to app
		apps = [self.app2vec_model.wv.vocab[intentApp[i]].index+1 for i in self.explicit_intent]

		# predicted vector
		vector_predict = self.bilstm.predict(apps,bilstm_model)

		# The predicted label
		predict_label = af_model.predict([vector_predict])

		# Get the candididate apps and avoid duplicate
		candidiates = list(filter(lambda x:x not in apps,list(set(self.af.label2app[predict_label[0]]))))

		pool = multiprocessing.Pool()
		manager = Manager()

		# For recording the semantic score
		shared_dict = manager.dict()

		for candidiate_id in range(len(candidiates)):

			# Calculate the semantic score
			pool.apply_async(calculateJob,args=(input_sen,candidiates[candidiate_id],p_data.app2des[candidiates[candidiate_id]]))

		pool.close()
		pool.join()

		# Sort by semantic score
		semantic_filter = sorted(shared_dict,key = shared_dict.get,reverse = True)

		result = self.p_data.checkClass(semantic_filter,5)

		return result
