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
4. We provide two mode. --> ANN-based model with Doc2Vec, Cluster-based model with Doc2Vec.

# Training Stage
> View Detail: https://github.com/TIS-JOEY/App2Vec_Python

## 1. Prepare App2Vec Training Data
App2Vec treats each app as a unit. And we use daily app usage data as our training data.
Of course, it's impossible to train the raw data directly.
So we provide the below function：

### Function: `App2Vec.csv2training_data`

Goal: Prepare the training data of App2Vec.

`raw_file_path` = The storage location of your raw training data (Currently, we only support the csv file).

The raw data is a csv file which should be like as below:
Each row is an app sequence which contains several apps.

| app sequence1 |
| --- |
| app sequence2 |
| app sequence3 |
| app sequence4 |
| app sequence5 |
| app sequence6 |

## 2. Train the app2vec model

### Function: `App2Vec.training_App2Vec`

Goal: Train the App2Vec model.

`app2vec_model_path` = The storage location of App2Vec model.

## 3. Train the ANN model

### Function `ANN`

Goal: Train the ANN model

`dim` = the Dimension of App2Vec.

`num_tree` = The number of trees of your ANN forest. More tress more accurate.

`ann_model_path` = The storage path of ANN model.

## 4. Train the Affinity Propagation model

Affinity Propagation is a unsupervised learning method which does not require the pre-defined number of clusters. It can automatically find a collection of objects which are representative of clusters and discover the number of clusters. In order to find the exemplars for each cluster, Affinity Propagation takes a set of pairwise similarities as input and passes the messages between these pairwise data objects. In this training stage, Affinity Propagation updates two matrices  and .  represent the responsibility of each object. A higher value for the  of object in cluster  means that object would be a better exemplar for cluster .  represent the availability of each object. A higher value for the  of object in cluster  means that object would be likely to belong to cluster . This updating is executed iteratively until convergence. Once convergence is achieved, exemplars of each cluster are generated. Affinity Propagation outputs the final clusters.

In this interface, we use  Scikit-Learn library to achieve it (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html).

### Function `affinity_propagation`

Goal: Train the Affinity Propagation model.

`app2vec_model_path` = The storage location of App2Vec model.

`af_model_path` = The storage location of Affinity Propagation model.

`prefer` = The preference of Affiniry Propagation model.

### Function `get_label2id`

Goal: Build the mapping between Affinity Propagation's labels and app sequences (Store in a object attribute which name is label2id).

`af_model_path` = The storage location of Affinity Propagation model.

```text
import app2vec.App2Vec

app2vec = App2Vec()

# Prepare the training data of App2Vec.
app2vec.csv2training_data(raw_file_path = '/Users/apple/Documents/raw_data.csv')

# Train the App2VApp2Vec model.
app2vec.training_App2Vec(model_path = '/Users/apple/Documents/app2vec.model')

# Train the ANN model.
app2vec.ANN(dim = 64,num_tree = 10000,app2vec_model_path = '/Users/apple/Documents/app2vec.model',ann_model_path = '/Users/apple/Documents/ANN.model')

# Train the affinity propagation model
app2vec.affinity_propagation(app2vec_model_path = '/Users/apple/Documents/app2vec.model',af_model_path = '/Users/apple/Documents/NewAFCluster.pkl',prefer = -30)

# Build the mapping between Affinity Propagation's labels and app sequences.
app2vec.get_label2id(af_model_path = '/Users/apple/Documents/AFCluster.pkl')
```
