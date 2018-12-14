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
#iterator class to create object passed to model training with the tag set to be the line number (only odd lines are processed as they are said by the user even lines are the responses)
class LabeledLineSentence(object):
	def __init__(self, filename):
		self.filename = filename
	def __iter__(self):
		for uid, line in enumerate(open(self.filename)):
			if (uid%2==1):
				words=line.lower().replace("-","").replace("?","").replace("'","").split()
				yield LabeledSentence(words, tags=[uid])
#compute model for each document (genre) separately)
for doc in documents:
	data = []
	with open(data_dir+doc, 'r') as f:
		data.append(f.read())
	#get iterator object, build vocab , train and save the model
	it = LabeledLineSentence(data_dir+doc)
	#initialize doc2vec model
	#min_count=minimum number of word appearances for a word to be in vocab
	#alpha= the learning rate
	#window for word2vec computation
	model = gensim.models.Doc2Vec(size=100,min_count=1,alpha=0.5,min_alpha=0.001,window=1)
	model.build_vocab(it)
	model.train(it,total_examples=model.corpus_count, epochs=100)
	model.save(model_dir+doc+".model")
#evaluate model accuracy over entire documents
true_pred=0
total_sentences=0
total_accuracy=0
for x in documents:
	true_pred=0
	total_sentences=0
	for uid, line in enumerate(open(data_dir+ x)):
		if (uid%2==1):
			total_sentences=total_sentences+1
			preprocessed = line.lower().replace("-","").replace("?","").replace("'","").split()
			tags=[uid]
			doc_model = Doc2Vec.load(model_dir+x+".model")
			utterance_doc_vector = doc_model.infer_vector(preprocessed)
			#get predicted sentence
			sims = doc_model.docvecs.most_similar([utterance_doc_vector]) 
			matched_sentence_index=sims[0][0]
			if (matched_sentence_index==uid):
				true_pred=true_pred+1
	accuracy=true_pred/total_sentences
	total_accuracy=accuracy+total_accuracy
	print("accuracy of model of document "+x+" is: "+str(accuracy))	
print("average model accuracy over all documents is: "+str(total_accuracy/len(documents)))	