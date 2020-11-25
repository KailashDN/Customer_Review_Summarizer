import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
import matplotlib.pyplot as plt
from config import *


class ReviewProcessing:
    def __init__(self, review_len=100000):
        self.filepath = datapath
        self.data = pd.read_csv(self.filepath, nrows=review_len)
        # self.contraction_mapping2 = open('contraction_mapping').read()
        self.contraction_mapping = cont_mapping

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

    def data_analysis(self):
        text_word_count = []
        summary_word_count = []
        fig = plt.figure(figsize=(18, 6))
        # populate the lists with sentence lengths
        for i in self.data['cleaned_text']:
            text_word_count.append(len(i.split()))

        for i in self.data['cleaned_summary']:
            summary_word_count.append(len(i.split()))

        length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(length_df['text'], 'b-')
        ax1.set_title("Text distribution")

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(length_df['summary'], 'r^')
        ax2.set_title("Summary distribution")

        # Save the full figure...
        fig.savefig('Analysis/distribution of the sequences.png')

        # fig = length_df.hist(bins=30)
        plt.show()

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
        self.data['cleaned_text'] = cleaned_text
        self.data['cleaned_summary'] = cleaned_summary

        self.data.replace('', np.nan, inplace=True)
        self.data.dropna(axis=0, inplace=True)

        # From analysis 94% of summary length is below 8 and we are trimming maximum length of text to 30
        self.data_analysis()
        # max_text_len = 30
        # max_summary_len = 8

        cleaned_text = np.array(self.data['cleaned_text'])
        cleaned_summary = np.array(self.data['cleaned_summary'])
        short_text = []
        short_summary = []

        for i in range(len(cleaned_text)):
            if len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len:
                short_text.append(cleaned_text[i])
                short_summary.append(cleaned_summary[i])

        df = pd.DataFrame({'text': short_text, 'summary': short_summary})

        # Adding START: sostok and END:eostok tokens
        df['summary'] = df['summary'].apply(lambda x: 'sostok ' + x + ' eostok')
        # print(df)
        return df


# pre_p = ReviewProcessing(100000)
# pre_p.pre_process()
