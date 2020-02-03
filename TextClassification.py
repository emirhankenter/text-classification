#In[]
#1-Import libraries
import tensorflow.compat.v1 as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import string
import gensim
import datetime
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import preprocessing

#In[]
#2-Check GPU is running
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#In[]
#3-Activating eager execution
tf.enable_eager_execution()
tf.executing_eagerly()

#In[]
#4-Defining path
path="/Users/emirh/Desktop/bitirme"

#In[]
#5-Reading dataset file
df = pd.DataFrame()
df = pd.read_csv(path+'/movie_data.csv', encoding='utf-8')
df.head(3)

#In[]
#6-Removing punctuations and stopwords
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
example_sent = "This is not a sample sentence, showing off the stop words filtration."
  
stop_words = set(stopwords.words('english')) 
#Remove critical stopwords
def removeStopWords(myArray):
    for criticalWord in myArray:
        stop_words.remove(criticalWord)
criticalWords = ["no","not","wasn't","isn't","against","don't","doesn't","aren't","hasn't","couldn't","hadn't","haven't"]
removeStopWords(criticalWords)

word_tokens = word_tokenize(example_sent) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = []
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence) 

#In[]
#7-Tokenize
review_lines = list()
lines = df['review'].values.tolist()

for line in lines:   
    tokens = word_tokenize(line)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words    
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)

#In[]
#8-Train word2vec model
EMBEDDING_DIM = 250
# train word2vec model
model = gensim.models.Word2Vec(sentences=review_lines, size=EMBEDDING_DIM, window=8, workers=5, min_count=20,negative=5,iter=20)
# vocab size
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

#In[]
#9-Save model in ASCII (word2vec)format
filename = path+'/imdb_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

#In[]
#10-Word embedding
embeddings_index = {}
f = open(os.path.join('', path+'/imdb_embedding_word2vec.txt'),  encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()
max_length = 150 # assuming documents are 500 words
#In[]
#11-Prepare train and test set
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# vectorize the text samples into a 2D integer tensor
tokenizer_obj = Tokenizer(num_words=50000)
#50000 most frequent words will be kept
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)

# pad sequences
word_index = tokenizer_obj.word_index
print('Found %s unique tokens.' % len(word_index))

review_pad = pad_sequences(sequences, maxlen=max_length)
sentiment =  df['sentiment'].values
print('Shape of review tensor:', review_pad.shape)
print('Shape of sentiment tensor:', sentiment.shape)

# split the data into a training set and a validation set
indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]

X_train_pad = review_pad[:40000]
y_train = sentiment[:40000]
X_eval_pad = review_pad[40001:45000]
y_eval = sentiment[40001:45000]
X_test_pad = review_pad[45001:]
y_test = sentiment[45001:]

#In[]
#Save tokenizer_obj
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
#In[]
#12
print('X_train_pad tensor:', X_train_pad[1])
print('y_train tensor:', y_train[1])

print('Shape of X_test_pad tensor:', X_test_pad.shape)
print('Shape of y_test tensor:', y_test.shape)

#In[]
#13-This is a lookup dictionary for embedding
EMBEDDING_DIM = 250
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#In[]
#14
print('Words = '+str(num_words))
print('Embedding matrix = '+str(len(+embedding_matrix)))

#In[]
#15 Creating model to train
from tensorflow.compat.v1 import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
filepath=path+"/best_model.pkl"
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)
es = EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', verbose=1, patience=200)
mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)
def createModelStructure():
    model = Sequential([
        embedding_layer,
        LSTM(units=64,dropout=0.2,recurrent_dropout=0.2),
        Dense(500,activation='relu',kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.2),
        Dense(300, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = createModelStructure()

print('Summary of the built model...')
print(model.summary())

#In[]
#16 - Train
print('Train...')

model.fit(X_train_pad, y_train, batch_size=512, epochs=20, validation_data=(X_eval_pad, y_eval), callbacks=[tensorboard_callback,es,mc])
#In[]
#17-Load model to new model
saved_model = createModelStructure()
saved_model.load_weights(filepath)
filename = 'finalized_model.pkl'
#In[]
#Save finalized_model
joblib.dump(saved_model, filename)
#In[]
#Load finalized_model
saved_model = joblib.load(path+'/'+filename)
#In[]
#18-Load Tensorboard extension
%load_ext tensorboard
#In[]
#19
from tensorboard import notebook
notebook.list()
#In[]
#20
notebook.display(port=6006, height=1000) 
#In[]
#21
%tensorboard --logdir logs

#In[]
#22 Test the model
print('Testing...')
score, acc = saved_model.evaluate(X_test_pad, y_test, batch_size=512)

print('Test score:', score)
print('Test accuracy:', acc)

print("Accuracy: {0:.2%}".format(acc))
#In[]
#23
y_pred = saved_model.predict(X_test_pad[:4999], batch_size=512)
y_pred=(y_pred>0.58)
#In[]
#24 - Output precision recall score
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test[:4999], y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
#In[]
#25 - Output confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test[:4999], y_pred.round())

print(cm)
#In[]
#26
from sklearn.metrics import classification_report

print(classification_report(y_test[:4999], y_pred.round()))

#In[]
#Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
#In[]
#27 - Let us test some  samples
test_sample_1 = "This movie is fantastic! I really like it because it is so good!"
test_sample_2 = "Good movie!"
test_sample_3 = "Maybe I like this movie."
test_sample_4 = "Not to my taste, will skip and watch another movie"
test_sample_5 = "if you like action, then this movie might be good for you."
test_sample_6 = "Bad movie!"
test_sample_7 = "Not a good movie!"
test_sample_8 = "This movie really sucks! Can I get my money back please?"
test_sample_9 = "Not bad"
test_sample_10 = "Great actors"
test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5, test_sample_6, test_sample_7, test_sample_8, test_sample_9, test_sample_10]

test_samples_tokens = tokenizer.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

#predict
y_pred = saved_model.predict(x=test_samples_tokens_pad)
y_pred = (y_pred>0.58)
#if y_pred is bigger than 0.58, it is positive, else negative
for i in y_pred:
    if i>0.58:
        print('Positive')
    else:
        print('Negative')
#In[]
#Sample query
my_test = ["wasn't"]
test = tokenizer.texts_to_sequences(my_test)
test_pad = pad_sequences(test, maxlen=max_length)
test_pred = saved_model.predict(x=test_pad)
test_pred = (test_pred>0.58)
print(test_pred[0][0])

# %%
