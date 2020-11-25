import pickle


def validate_summarizer():

    # loading
    with open('saved_model_data/x_tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)
    with open('saved_model_data/y_tokenizer.pickle', 'rb') as handle:
        y_tokenizer = pickle.load(handle)

    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index
