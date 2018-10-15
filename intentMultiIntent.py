import re
from aip import AipNlp
from watson_developer_cloud import ConversationV1
import os
import numpy as np
from multiprocessing import Process, Manager,Lock
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


class EMIP:
	'''
	Process explict multi-intent.
	'''
	def __init__(self,conj,workspace_id):
		manager = Manager()
		self.baidu_nlp = None
		self.watson_nlp = None
		self.conjunction = []
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

	def getSaver(self):
		return self.intent_saver








