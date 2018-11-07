# The-multi-intent-detection-framework-of-Intelligent-Virtual-Assistant
I construct an Intelligent Virtual Assistant which can process multi-intent in a dialog turn. (Use App2Vec, Doc2Vec, ANN, Affinity Propagation)

# Motivation
In recent year, research interest in Intelligent Virtual Assistant (IVA) has soared in the world. However, current IVA is usually limited to specific domain and only handle a single intent per time. Current IVAs ignore the potential relationship between application services. 
However, people’s intents are usually complex and require several different applications to meet.
If users want to fulfill a complex task, they need to interact with IVA several times. It is time-consuming and troublesome in this way, especially for elder users or handicapped.
Obviously, it is important to find a way to break these limitations.
For this reason, I construct an Intelligent Virtual Assistant which can process multi-intent in a dialog turn. In this API, people’s intents are categorized into two types which are the explicit intent and the implicit intent.

# Features
This multi-intent Intelligent Virtual Assistant has several features are shown as below:
1. It can handle both the users’ explicit multi-intent and the users’ implicit multi-intent in a dialogue turn.
2. Chinese Natural Language Processing
3. It can be build even if there are lack of multi-intent-labeled training data.
4. We provide four modes. --> ANN-based model,Cluster-based model,ANN-based model with Doc2Vec, Cluster-based model with Doc2Vec.

# Training Stage
> View Detail: https://github.com/TIS-JOEY/Implicit-Intent-Inference-Model.git
> just follow the readme

# Predict stage
> You should configure Google NLP API on your computer. View more: https://cloud.google.com/natural-language/
## Installation

` git clone https://github.com/TIS-JOEY/The-multi-intent-detection-framework-of-Intelligent-Virtual-Assistant `


In this case, we can see the usage as below.

## Usage
```text
import explicit_multiIntent

# Processing the explict multi-intent
input_text = '今天可以去蘆洲吃晚餐然後去陽明山看夜景嗎'
emip = multiIntent.EMIP(conj = workspace_id = '123')

#Baidu NLP API, view more https://cloud.baidu.com/doc/NLP/NLP-Python-SDK.html
emip.baidu_api(API_ID = '123',API_KEY = '123',SECRET_KEY = '123)

#Watson NLP API, view more https://www.ibm.com/watson/developer/
emip.watson_api(usr_name = 'joey', passwd = '123')
emip.global_process(input_text)

#Explicit Multi Intent Result
explicit_multi_intent = emip.getIntent()


# Process the implicit multi-intent
data = {}
with open(r'Training/data/Model/app2des.json','r',encoding = 'utf-8') as f:
	data = json.load(f)

# Load the mapping of explict intent and apps
mapping = {}
with open(r'Training/data/Model/app_mapping.json','r',encoding = 'utf-8') as f:
	mapping = json.load(f)

imip = IMIP(explicit_intent = explicit_multi_intent,intentApp = mapping,app2vec_model_path = r'Training/data/Model/app2vec.model',ann_model_path = r'Training/data/Model/ann_model.ann',af_model_path = r'Training/data/Model/af_model.pkl',app2des = data)

# The parameter:
# model : ANN or AF
# ranker : mv, mf or doc
# lstm : True or False
print(imip.query(HanziConv.toSimplified(input_text),model = 'ANN',ranker = 'doc',lstm = False))
```
