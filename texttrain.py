import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


# LOAD DATA
df = pd.read_csv("IMDB Dataset.csv")

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

texts = df['review'].values
labels = df['sentiment'].values

# SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# TOKENIZER
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

train_seq = tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)

train_pad = pad_sequences(train_seq, maxlen=128)
test_pad = pad_sequences(test_seq, maxlen=128)

# MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_pad, y_train, epochs=2, validation_data=(test_pad, y_test))

model.save("text_model.h5")
print("Text model trained & saved")