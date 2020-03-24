import tensorflow as tf
import pandas as pd
import numpy as np
from zipfile import ZipFile
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from nltk import word_tokenize
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model
import time
from LSTM_Node import LSTM_Node, FC_Node, Network_Tree, LSTM_Tree, Embedding_Node
import spacy

#So I favorited the author of this models git page. It has a much more in depth implementation it looks like. 
#So going through that at some point and deciphering it would be a good thing I think.


#Goal of this deep learning model is to take in a headline and short description os some article and output the articles category.
start = time.time()

#Get a set of filler words from the english langauge using nltk stopwords package. We will remove these words from our text during the cleanup step
#Note we should probably generate our own set for future tasks (That is on top of some existing set, combine both) 
stop_words_ = set(stopwords.words('english'))

#Additional task specific words we would like to eliminate during the cleanup step
my_sw = ['make', 'amp',  'news','new' ,'time', 'u','s', 'photos',  'get', 'say']

#Our lemitizer we use one from the nltk package
wn = WordNetLemmatizer()

#still not really sure what this is doing
def black_txt(token):
        return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2 and token not in my_sw


#Clean up puncatuation and capilization then lemantize our text
def clean_txt(text):
	clean_text = []
	clean_text2 = []
	text = re.sub("'", "", text)
	text = re.sub("\\d|\\W+", " ", text)
	clean_text = [wn.lemmatize(word, pos = "v") for word in nltk.word_tokenize(text.lower()) if black_txt(word)]
	clean_text2 = [word for word in clean_text if black_txt(word)]
	return " ".join(clean_text2)


#Determining text polarity (positive/negative spectrum) meta-data collumn
def polarity_txt(text):
      return TextBlob(text).sentiment[0]

#Determining text subjectivity -> meta data collumn
def subj_txt(text):
      return  TextBlob(text).sentiment[1]

#len text mata-data -> not sure why we are doing this atm? -> need to figure this out
def len_text(text):
    if len(text.split())>0:
        return len(set(clean_txt(text).split()))/ len(text.split())
    else:
        return 0

#Tokenizer (converting raw text into some list based format for further processing)
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes



#name is a work in progress. But the gist of what this class should do is allow you to pass some set of hyper parameters and build a lstm network based on those.
#Additionaly it should have a bunch of helper methods that eval and do other fun things. This class should eventually morph into a parent class with generic imp.
#of a bunch of things and have daughter implement whatever application specefic stuff they need! -> I really like this idea.
class LSTM_Eval():

    def __init__(self, num_elm, emb_len, rand_seed, max_seq_len, batch_size, epochs, net):  

        #we are trying to speed up processing time while we work on the code so we are going to restrict the number of elements
        self.num_elm = num_elm

        #Read in our dataset
        self.data = self.read_json_data(num_elm)

        self.net = net
        self.rand_seed = rand_seed
        self.emb_len = emb_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.epochs = epochs

        self.run_pre_proc_steps()
        self.net_setup()
   
    #run all the pre_processing steps for out network
    def run_pre_proc_steps(self):    
        self.catg_clean()
        self.create_input()
        self.create_meta_data()
        self.get_pretrained_vector_model()

    #set up tf varaibles for out network
    def net_setup(self):
        self.gen_input_param()
        self.create_vectorizer()
        self.create_train_and_test_param()
        self.create_embeddings()

    #Merge similar catogries. Note this is now super specefic to our news example should eventually make this very generic.
    def catg_clean(self):

        #merge similar category WORDPOST and THE WORDPOST
        self.data.category = self.data.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

    #Create the input data for out network. This is very specefic atm should make more generic in future iterations.
    def create_input(self):

        #We will combine the categories of headline and short description into a new category text and use that as our prediction input
        self.data['text'] = self.data['headline'] + " " + self.data['short_description']

    
    #Create meta_data for our net. Super specefic make more general in future. Note not all networks neccisarly need this so it shouldn't be a forced overload method. 
    def create_meta_data(self):
        #Creating meta data collumns using textblob and pandas -> This needs to be merged with our vecotrized elements to be passed to network
        self.data['polarity'] = self.data['text'].apply(polarity_txt)
        self.data['subjectivity'] = self.data['text'].apply(subj_txt)
        self.data['len'] = self.data['text'].apply(len_text)


    #read in json data to a pandas structure and assign it to class data varaible. Prune to number of data elemnts to size num_elements.
    def read_json_data(self, num_elements):
        data = pd.read_json("News_Category_Dataset_v2.json", lines = True)
    
        if num_elements < len(data):
            data = data[:num_elements]
    
        return data
    
    #google vector representaion pretrained of english lang.
    def get_pretrained_vector_model(self):
        self.nlp = spacy.load('en_core_web_lg')


    #Create the input paramaters for out network. One hot label encoding and data input parameters.
    def gen_input_param(self):

        #our predictive variable (network input)
        self.X = self.data['text']

        #training varaible (actual results -> used for training(optimizing)  and testing network) 
        self.y = self.data['category']

        #Encoding our catogories -> numerically representing them
        encoder = LabelEncoder()

        #take our categorical data and convert to numerical values
        self.y = encoder.fit_transform(self.y)

        #tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32'), 
        #y: class converter to matrix (0-num_classes), num_classes: number of catergories, dtype: data type
        #Creates one hot representation
        self.Y = np_utils.to_categorical(self.y)


    #create the vectorized for out network
    def create_vectorizer(self):

        #Create the tf-idf vector
        self.vectorizer = TfidfVectorizer( min_df =3, max_df=0.2, max_features=None, 
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words = None, preprocessor=clean_txt)



    #create the tf paramters for our testing and training 
    def create_train_and_test_param(self):

        #Split our input and prediction varaible into training and testing data -> maybe implemnt in proj. as we mod seed_params
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state = self.rand_seed, stratify = self.y)

        #Learn vocabulary and idf from training set.add
        self.vectorizer.fit(self.x_train)

        #Create a python dict that maps feautre name to feature index
        self.word2idx = {word: idx for idx, word in enumerate(self.vectorizer.get_feature_names())}

        #Return a function that splits a string into a sequence of tokens
        tokenize = self.vectorizer.build_tokenizer()

        #Return a function to preprocess the text before tokenization
        preprocess = self.vectorizer.build_preprocessor()

        #Pre-process our data using skleaen and then tokenize that pre-proccesed data. 
        #Then create a list of lists (where each sub-list represents some subsection of our text data as dictcated by test train and split, we are looking only at training data here).
        #Note that each element in this list will be some numerical value that is equivalent to the mapped value from our word2idx dictionary.
        self.X_train_sequences = [to_sequence(tokenize, preprocess, self.word2idx, x) for x in self.x_train]

        self.df_cat_train = self.data.iloc[self.x_train.index][['polarity', 'subjectivity', 'len']]
        self.df_cat_test = self.data.iloc[self.x_test.index][['polarity', 'subjectivity', 'len']]

        #Get number of features 
        N_FEATURES = len(self.vectorizer.get_feature_names())

        #Crop or pad sequences to max sequence length, padding value is num_features(not sure why we are doing that over 0?) -> feels like a var we can play with and optimize
        self.X_train_sequences = pad_sequences(self.X_train_sequences, maxlen = self.max_seq_len, value=N_FEATURES)

        #??? miscopy on his end maybe not sure why this is here looks redudant but we can play with it to find out, perfomance comparison and what not.
        self.X_test_sequences = [to_sequence(tokenize, preprocess, self.word2idx, x) for x in self.x_test]
        self.X_test_sequences = pad_sequences(self.X_test_sequences, maxlen=self.max_seq_len, value=N_FEATURES)



    def create_embeddings(self):
        #create a zero matrix of size (num_features + 1, EMBEDDINGS_LEN) -> again not sure what this if for
        self.embeddings_index = np.zeros((len(self.vectorizer.get_feature_names()) + 1, self.emb_len))

        #create an embedding index with list of vector representations of words i think?
        for word, idx in self.word2idx.items():
            try:
                embedding = self.nlp.vocab[word].vector
                self.embeddings_index[idx] = embedding
            except:
                pass
                                              

    #train our network
    def train_net(self):
        text_data = Input(shape=(self.max_seq_len,), name='text')
        meta_data = Input(shape=(3,), name = 'meta')

        
        x = Embedding(len(self.vectorizer.get_feature_names()) + 1,
                                self.emb_len,  # Embedding size
                                weights=[self.embeddings_index],
                                input_length=self.max_seq_len,
                                trainable=False)(text_data)

        #So there is a Q on where emebedding should be defined. Here where all fxns already exists or somewhere else for abstraction.
        term_tens = [x, meta_data]
        outp = self.net.build_net(term_tens)

        self.AttentionLSTM = Model(inputs=[text_data, meta_data ], outputs=outp)
        self.AttentionLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.AttentionLSTM.summary()
        self.AttentionLSTM.fit([self.X_train_sequences, self.df_cat_train], self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, validation_split=0.1)

    #test our network
    def test_net(self):
        scores = self.AttentionLSTM.evaluate([self.X_test_sequences, self.df_cat_test],self.y_test, verbose=1)
        print("Accuracy:", scores[1])  # 

    #train and test our network. Display results.
    def call(self):
        self.train_net()
        self.test_net()


"""
#number of data parameter we prune to -> should be set to max when network is finished but mostly used as a way to speed up testing/training when still playing with network
#bassically this prevent you from waiting an hour for results when you are testing something other then network perfomance/debugging.
NUMBER_ELEMENTS = 5000

#random seed value for something to cluster around proabably
seed = 40

#output size of first layer
EMBEDDINGS_LEN = 300

#Compute the max lenght of a text
MAX_SEQ_LENGHT=60

epochs = 8
batch_size = 128

net = LSTM_Eval(NUMBER_ELEMENTS, EMBEDDINGS_LEN, seed, MAX_SEQ_LENGHT, batch_size, epochs)
net.call()

#print("With {0} elements total run time is: {1}".format(Num_elm, stop - start))

"""
