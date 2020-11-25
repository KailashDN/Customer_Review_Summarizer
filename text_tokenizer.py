from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
import numpy as np


class TextTokenizer:
    def __init__(self):
        self.max_text_len = 30
        self.max_summary_len = 8

    def split_data(df):
        return train_test_split(np.array(df['text']), np.array(df['summary']), test_size=0.1, random_state=0, shuffle=True)

    def rarewords(self, tokenizer):
        thresh = 4

        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if (value < thresh):
                cnt = cnt + 1
                freq = freq + value

        print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
        print("Total Coverage of rare words:", (freq / tot_freq) * 100)
        return cnt, tot_cnt, freq, tot_freq

    def remove_tags(self, x_val, y_val):
        ind = []
        for i in range(len(y_val)):
            cnt = 0
            for j in y_val[i]:
                if j != 0:
                    cnt = cnt + 1
            if (cnt == 2):
                ind.append(i)

        y_val = np.delete(y_val, ind, axis=0)
        x_val = np.delete(x_val, ind, axis=0)
        return x_val, y_val

    def tokenizer(self, df):
        # prepare a tokenizer for reviews on training data
        x_tr, x_val, y_tr, y_val = self.split_data(df)
        x_tokenizer = Tokenizer()
        x_tokenizer.fit_on_texts(list(x_tr))
        print("Proportion of Rarewords and its coverage in entire Text:")
        cnt, tot_cnt, freq, tot_freq = self.rarewords(x_tokenizer)
        # prepare a tokenizer for reviews on training data
        x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        x_tokenizer.fit_on_texts(list(x_tr))

        # convert text sequences into integer sequences
        x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
        x_val_seq = x_tokenizer.texts_to_sequences(x_val)

        # padding zero upto maximum length
        x_tr = pad_sequences(x_tr_seq, maxlen=self.max_text_len, padding='post')
        x_val = pad_sequences(x_val_seq, maxlen=self.max_text_len, padding='post')

        # size of vocabulary ( +1 for padding token)
        x_voc = x_tokenizer.num_words + 1

        # prepare a tokenizer for reviews on training data
        y_tokenizer = Tokenizer()
        y_tokenizer.fit_on_texts(list(y_tr))
        print("Proportion of Rarewords and its coverage in entire Summary:")
        self.rarewords(y_tokenizer)
        # prepare a tokenizer for reviews on training data
        y_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        y_tokenizer.fit_on_texts(list(y_tr))

        # convert text sequences into integer sequences
        y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
        y_val_seq = y_tokenizer.texts_to_sequences(y_val)

        # padding zero upto maximum length
        y_tr = pad_sequences(y_tr_seq, maxlen=self.max_summary_len, padding='post')
        y_val = pad_sequences(y_val_seq, maxlen=self.max_summary_len, padding='post')

        # size of vocabulary
        y_voc = y_tokenizer.num_words + 1

        x_tr, y_tr = self.remove_tags(x_tr, y_tr)
        x_val, y_val = self.remove_tags(x_val, y_val)

        print(f"{x_tr}\n{x_val}\n{x_voc}")
        print(f"{y_tr}\n{y_val}\n{y_voc}")
