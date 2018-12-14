from os import listdir
from os.path import isfile, join
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim.models.doc2vec import LabeledSentence
from gensim.parsing.preprocessing import preprocess_documents

data_dir="data/"
model_dir="model/"
#get list of files in data directory representing genres
documents = []
documents = [f for f in listdir(data_dir) if f.endswith('.yml')]
#get sentences in the genre files to be passed to LabeledLineSentence iterator
data = []	
for doc in documents:

	with open(data_dir+ doc, 'r') as f:

		data.append(f.read())

labelled_data = []	
for doc in documents:

	with open(data_dir+ doc, 'r') as f:

		labelled_data.append(f.read())
		
#iterator class to create object passed to model training with tag set to be the file name which is the genre
class LabeledLineSentence(object):
	def __init__(self, doc_list, labels_list):
	   self.labels_list = labels_list
	   self.doc_list = doc_list
	def __iter__(self):
		for idx, doc in enumerate(self.doc_list):
			words=doc.lower().replace("-","").replace("?","").replace("'","").split()
			#remove stop words , punctuation and numbers
			preprocess=preprocess_documents(words)
			#convert unicode to ascii string
			preprocessed=[str(i) for i in preprocess]
			yield LabeledSentence(preprocessed,tags=[self.labels_list[idx]])
#get iterator object, build vocab , train and save the model
it = LabeledLineSentence(data, documents)
#initialize doc2vec model
#min_count=minimum number of word appearances for a word to be in vocab
#alpha= the learning rate
#min_alpha= minimum learning rate
#dm_concat= concatenate word vectors rather than average
model = gensim.models.Doc2Vec(size=200,min_count=1,alpha=0.025,min_alpha=0.001,dm_concat=1)#,window=7)#,train_words=True,learn_doctags =True,learn_words =True)
model.build_vocab(it)
model.train(it,total_examples=model.corpus_count, epochs=50)#best is 48
model.save(model_dir+"genre.model")


#evaluate model accuracy over entire documents
true_pred=0
for x in documents:
	with open(data_dir+ x, 'r') as f:
		utterance=f.read()
		preprocessed = utterance.lower().replace("-","").replace("?","").replace("'","").split()
		preprocess=preprocess_documents(preprocessed)
		preprocessed=[str(i) for i in preprocess]
		utterance_genre_vector = model.infer_vector(preprocessed)
		sims = model.docvecs.most_similar([utterance_genre_vector])
		if (sims[0][0]==x):
			true_pred=true_pred+1
accuracy=true_pred/len(documents)
print("model accuracy over entire document is: "+str(accuracy))	
#evaluate model accuracy over entire documents
true_pred=0
total_sentences=0
for x in documents:
	with open(data_dir+ x, 'r') as f:
		document=f.read().splitlines()
		for line in document:
			total_sentences=total_sentences+1
			utterance=line
			pre = utterance.lower().replace("-","").replace("?","").replace("'","").split()
			preprocess=preprocess_documents(pre)
			preprocessed=[str(i) for i in preprocess]
			utterance_genre_vector = model.infer_vector(preprocessed)
			sims = model.docvecs.most_similar([utterance_genre_vector])
			if (sims[0][0]==x):
				true_pred=true_pred+1
accuracy=true_pred/total_sentences
print("model accuracy over sentences is: "+str(accuracy))	