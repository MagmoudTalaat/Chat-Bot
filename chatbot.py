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
genre_model_dir="model/genre.model"
#genre model loader so that the server loads on cold start to decrease response time
def load_genre_model():
	#load genre classification model and predict target document(genre)
	genre_model = Doc2Vec.load(genre_model_dir)
	return genre_model

def get_response(utterance,genre_model):
	#preprocess utterence to remove unwanted characters and convert to lower case
	pre = utterance.lower().replace("-","").replace("?","").replace("'","").split()
	preprocess=preprocess_documents(pre)
	preprocessed=[str(i) for i in preprocess]
	#get doc2vec representation for utterance with respect to genre classifier model
	utterance_genre_vector = genre_model.infer_vector(preprocessed)
	#get predicted document(genre)
	sims = genre_model.docvecs.most_similar([utterance_genre_vector])
	matched_genre=sims[0][0]
	#load in document sentence classification model and predict target sentence
	doc_model = Doc2Vec.load(model_dir+matched_genre+".model")
	#get doc2vec representation for utterance with respect to  sentence classification model
	utterance_doc_vector = doc_model.infer_vector(preprocessed)
	#get predicted sentence
	sims = doc_model.docvecs.most_similar([utterance_doc_vector]) 
	matched_sentence_index=sims[0][0]
	#get corresponding response to that sentence and return it
	f=open(data_dir+matched_genre)
	lines=f.readlines()
	response=""
	if (matched_sentence_index+1<len(lines)):
		response= lines[matched_sentence_index+1]
	else:
		response=lines[matched_sentence_index]
	return response.replace("-","")

	
