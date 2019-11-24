import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import os
import pandas as pd
import re
import string
import sys
def csv_search():
    '''
    looks for .csvs exclusively in the current working directory
    and returns them to main
    Args:
        None
    Returns:
        dir_walk(list): a list of available .csv files
    '''

    dir_walk = [file for root,dirs,files in os.walk('.')
                for file in files if file.endswith('csv')]
    line_count = 0
    for i in dir_walk:
        line_count += 1
        print(str(line_count) + ' ' + i)
    return dir_walk


def csv_select(dir_walk):
    print("Please make a selection\n Pressing Enter will attempt to use all csvs in the current directory\nIn order to make specific selections use the format '1,3,4' or '2-4'")
    selection = input("SELECTION: ")
    if ',' in selection:
        choices = selection.split(',')
        print(choices)

def find_content(tgt_files):
    '''
    takes a list of files in as input and then reads
    them, producing two things, 'details' which is the date,
    author, publisher and content of the article and 'content'
    which is the story for each articles

    Args:
        tgt_files(string):.csv files that are in the current directory
    Returns:
        details(generator object): see above
        content(generator object): see above
    '''
    fields = ['content']
    file = [file for file in tgt_files]
    df = pd.read_csv('articles1.csv', skipinitialspace=True, usecols=fields)
    for i in df['content']:
        light_analysis(i)
    read_csv = (pd.read_csv(csv_file, skipinitialspace=True, usecols=fields)
                for csv_file in file)
    details = (details[['date','publication','author','content']]
                for details in read_csv if "," not in details)
    return


def light_analysis(text):
    '''
    Takes content, like a news story, as input
    then looks at frequency analysis and returns them to
    the terminal.

    Args:
        text(string): content generator from find_content
    Returns:
        maybe a list of the 25 most_common words
    '''
    not_letters = ["'",'"',',','.','?','!',':',';']
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stripped = remove_stopwords(word_tokens, stop_words)
    freq = FreqDist(stripped)
    return freq


def remove_stopwords(word_tokens, stop_words):
    stripped_text = [word for word in word_tokens if not word in stop_words]
    stripped_text = []
    for word in word_tokens:
        if word not in stop_words and word.isalpha():
            stripped_text.append(word)
    return stripped_text


def argparser(passed_args=None):
    parser = argparse.ArgumentParser(description = 'performs various forms of Natural Language Processing on given .csv files')
    args = parser.parse_args()
    return args


def main(passed_args=None):
    tgt_files = csv_search()
    selector = csv_select(tgt_files)
    find_content(tgt_files)


if __name__ == '__main__':
    main()
