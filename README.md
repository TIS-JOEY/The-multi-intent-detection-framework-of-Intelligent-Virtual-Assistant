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
> View Detail: https://github.com/TIS-JOEY/Build-and-Evaluate-App2Vec-ANN-Affinity-Porpagation

## Installation
` git clone https://github.com/TIS-JOEY/Build-and-Evaluate-App2Vec-ANN-Affinity-Porpagation.git `
After cloning, you should rename the project because we want to import this module.
For example: You can rename it to MIP_model.

In this case, we can see the usage as below.

## Usage
```text
import MIP_model.app2vec.App2Vec
import MIP_model.AF.AF
import MIP_model.ann.ANN
import MIP_model.processData.processData


# Prepare the data of App2Vec
p_data = processData(mapping_path = 'mapping.csv')
p_data.csv2App2Vec_training_data(raw_file_path = 'raw_data.csv')
p_data.save(write_file_path = 'training_data.txt')



# Train the app2vec model
app2vec = App2Vec()
app2vec.load_training_data(raw_file_path = 'training_data.txt')
app2vec.training_App2Vec(app2vec_model_path = 'app2vec.model')



# Prepare the data for evaluating App2Vec
X,y = p_data.csv2evaluate_App2Vec_training_data(raw_file_path = 'app2vec_evaluate_raw_data.csv')



# Evaluate the app2vec model
app2vec = App2Vec()
app2vec.grid_app2vec(X = X,y = y,app2vec_model_path = 'app2vec.model')
app2vec.show_app2vec(app2vec_model_path = 'app2vec.model')



# Train the ANN model
ann = ANN(app2vec_model_path = 'app2vec.model')
ann.train_ANN(dim = 90,num_tree = 10000,,ann_model_path = 'ann.model')



# Prepare the data for evaluating ANN
X,y = p_data.csv2evaluate_ANN_training_data(raw_file_path = 'raw_data.csv')



# Evaluate the ANN model
ann.evaluate_ann(X = X,y = y,dim = 90,app2vec_model_path = 'app2vec.model',ann_model_path = 'ann.model')



# Train the Affinity Propagation model
training_data = app2vec.load_training_data(raw_file_path = 'training_data.txt')
AF_model = AF(app2vec_model_path = 'app2vec.model',training_data = training_data)
AF_model.affinity_propagation(af_model_path = 'NewAFCluster.pkl',prefer = -30)



# Build the mapping between Affinity Propagation's labels and app sequences.
label2id = app2vec.get_label2id(af_model_path = 'AFCluster.pkl')
```

# Predict stage
> You should configure Google NLP API on your computer. View more: https://cloud.google.com/natural-language/
## Installation
` git clone https://github.com/TIS-JOEY/The-multi-intent-detection-framework-of-Intelligent-Virtual-Assistant `

After cloning, you should rename the project because we want to import this module.
For example: You can rename it to MIP_Predict.

In this case, we can see the usage as below.

## Usage
```text
import MIP_Predict.multiIntent

# Processing the explict multi-intent
input_text = '今天可以去蘆洲吃晚餐然後去陽明山看夜景嗎'
emip = multiIntent.EMIP(conj = workspace_id = '123')

#Baidu NLP API, view more https://cloud.baidu.com/doc/NLP/NLP-Python-SDK.html
emip.baidu_api(API_ID = '123',API_KEY = '123',SECRET_KEY = '123)

#Watson NLP API, view more https://www.ibm.com/watson/developer/
emip.watson_api(usr_name = 'joey', passwd = '123')
emip.global_process(input_text)

#Result
explicit_multi_intent = emip.getIntent()

# Processing the implicit multi-intent
intentApp = 'The mapping of app and intent'
des = 'The mapping of app and its description'

imip = multiIntent.IMIP(app2vec = App2Vec,explict_intent = explicit_multi_intent,intentApp = intentApp,app,app2vec_model_path = 'app2vec.model',ANN_model_path = 'ann.model',dim = 90,des = des,af_model_path = 'AFCluster.pkl',label2id = label2id)

ANN_based_model_result = imip.ANN_process()
ANN_based_model_with_Doc2Vec_result = imip.doc2vec_process(input_text)
imip.renew()
Cluster_based_model_result = imip.af_process()
Cluster_based_model_with_Doc2Vec_result = imip.af_doc_process(input_text)
```
