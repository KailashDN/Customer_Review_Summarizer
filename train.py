from tensorflow.keras.callbacks import EarlyStopping
from pre_process import ReviewProcessing
from text_tokenizer import TextTokenizer
from model_LSTM import model_LSTM
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def plot_results(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig("Analysis/train_test_accuracy.png")
    plt.show()


def train_model():
    rp = ReviewProcessing(100000)
    df = rp.pre_process()
    tkn = TextTokenizer(df)
    x_tr, x_val, x_voc, y_tr, y_val, y_voc, y_tokenizer, x_tokenizer = tkn.tokenizer()
    model = model_LSTM(x_voc, y_voc)

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=7, patience=4)

    history = model.fit([x_tr, y_tr[:, :-1]],
                        y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],
                        epochs=100,
                        callbacks=[es],
                        batch_size=128,
                        validation_data=([x_val, y_val[:, :-1]],
                                         y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))
    plot_results(history)
    model.save('saved_model_data/lstm_model.h5')

train_model()
