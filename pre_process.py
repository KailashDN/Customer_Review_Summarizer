import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

import re


class ReviewProcessing:
    def __init__(self, review_len=100000):
        self.filepath= "Dataset/Reviews.csv"
        self.data = pd.read_csv(self.filepath, nrows=review_len)
        # self.contraction_mapping2 = open('contraction_mapping').read()
        self.contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                                    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                                    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                                    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                                    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                                    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                                    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                                    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                                    "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                                    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                                    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                                    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                                    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                                    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                                    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                                    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                                    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                                    "you're": "you are", "you've": "you have"}

        self.stop_words = set(stopwords.words('english'))

    def text_cleaner(self, text, num):
        newString = text.lower()
        newString = BeautifulSoup(newString, "lxml").text
        newString = re.sub(r'\([^)]*\)', '', newString)
        newString = re.sub('"', '', newString)
        newString = ' '.join([self.contraction_mapping[t] if t in self.contraction_mapping else t for t in newString.split(" ")])
        newString = re.sub(r"'s\b", "", newString)
        newString = re.sub("[^a-zA-Z]", " ", newString)
        newString = re.sub('[m]{2,}', 'mm', newString)
        if num == 0:
            tokens = [w for w in newString.split() if not w in self.stop_words]
        else:
            tokens = newString.split()
        long_words = []
        for i in tokens:
            if len(i) > 1:  # removing short word
                long_words.append(i)
        return (" ".join(long_words)).strip()

    def pre_process(self):
        # remove null and duplicates
        self.data.drop_duplicates(subset=['Text'], inplace=True)  # dropping duplicates
        self.data.dropna(axis=0, inplace=True)  # dropping na

        cleaned_text = []
        for t in self.data['Text']:
            cleaned_text.append(self.text_cleaner(t, 0))

        cleaned_summary = []
        for t in self.data['Summary']:
            cleaned_summary.append(self.text_cleaner(t, 1))

        # print(cleaned_text[:5])
        # print("\n", cleaned_summary[:5])

        return cleaned_text, cleaned_summary


# pre_p = ReviewProcessing(100000)
# pre_p.pre_process()
