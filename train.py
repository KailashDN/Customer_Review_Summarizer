from pre_process import ReviewProcessing
from text_tokenizer import TextTokenizer
from model_LSTM import model_LSTM
from config import *

from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class TrainEvaluate:
    def __init__(self):
        self.rp = ReviewProcessing(100000)
        self.df = self.rp.pre_process()
        self.tkn = TextTokenizer(self.df)
        self.x_tr, self.x_val, self.x_voc, self.y_tr, self.y_val, self.y_voc, self.x_tokenizer, self.y_tokenizer = self.tkn.tokenizer()
        self.model, self.encoder_model, self.decoder_model = model_LSTM(self.x_voc, self.y_voc)
        self.reverse_target_word_index = self.y_tokenizer.index_word
        self.reverse_source_word_index = self.x_tokenizer.index_word
        self.target_word_index = self.y_tokenizer.word_index

    def plot_results(self, history):
        plt.clf()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig("Analysis/train_test_accuracy.png")
        plt.show()

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = self.target_word_index['sostok']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:

            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            if (sampled_token != 'eostok'):
                decoded_sentence += ' ' + sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence

    def seq2summary(self, input_seq):
        newString=''
        for i in input_seq:
            if(i!=0 and i!=self.target_word_index['sostok']) and i!=self.target_word_index['eostok']:
                newString=newString+self.reverse_target_word_index[i]+' '
        return newString

    def seq2text(self, input_seq):
        newString=''
        for i in input_seq:
            if(i!=0):
                newString=newString+self.reverse_source_word_index[i]+' '
        return newString

    def train_model(self):
        # rp = ReviewProcessing(100000)
        # df = rp.pre_process()
        # tkn = TextTokenizer(df)
        # x_tr, x_val, x_voc, y_tr, y_val, y_voc = tkn.tokenizer()
        # model = model_LSTM(x_voc, y_voc)

        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=7, patience=4)

        history = self.model.fit([self.x_tr, self.y_tr[:, :-1]],
                            self.y_tr.reshape(self.y_tr.shape[0], self.y_tr.shape[1], 1)[:, 1:],
                            epochs=100,
                            callbacks=[es],
                            batch_size=256,
                            validation_data=([self.x_val, self.y_val[:, :-1]],
                                             self.y_val.reshape(self.y_val.shape[0], self.y_val.shape[1], 1)[:, 1:]))
        self.plot_results(history)
        self.model.save('saved_model_data/lstm_model.h5')

        for i in range(0, 100):
            print("Review:", self.seq2text(self.x_tr[i]))
            print("Original summary:", self.seq2summary(self.y_tr[i]))
            print("Predicted summary:", self.decode_sequence(self.x_tr[i].reshape(1, max_text_len)))
            print("\n")


trn = TrainEvaluate()
trn.train_model()
