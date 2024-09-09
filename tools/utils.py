from keras.preprocessing import sequence
from keras.utils.data_utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

class Preprocessing:
    def __init__(self, args):
        self.data = 'c:/Users/AMEN/Desktop/FeelFree/model/data/data.csv'
        self.max_len = args.max_len
        self.max_words = args.max_words
        self.test_size = args.test_size

    def loadData(self):
        df = pd.read_csv(self.data)
        df.dropna(inplace=True)
        X = df['clean_text'].values
        Y = df['category'].values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=42)

    def prepareTokens(self):
        self.tokens = Tokenizer(num_words=self.max_words)
        self.tokens.fit_on_texts(self.x_train)

    def text2sequence(self, x):
        sequences = self.tokens.texts_to_sequences(x)
        return pad_sequences(sequences, maxlen=self.max_len)
