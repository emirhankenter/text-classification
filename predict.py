#In[]
import pickle
from sklearn.externals import joblib
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#In[]
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
#In[]
path="/Users/emirh/Desktop/bitirme/finalized_model.pkl"
saved_model = joblib.load(path)
#In[]
#27
#Let us test some  samples
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
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=150)

#predict
y_pred = saved_model.predict(x=test_samples_tokens_pad)
#y_pred = (y_pred>0.58)
print(y_pred)
# for i in y_pred:
#     if i>0.54:
#         print('Positive')
#     else:
#         print('Negative')


# %%
