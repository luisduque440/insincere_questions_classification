import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize

def get_good_and_bad_words(df, num_samples=100):
    """ Given a data frame with a column 'question_text' and another column named 'target'
    Finds the 'good words' (i.e. words that are more likely appear when target=0) and the 
    'bad words' (i.e. words that are more likely appear when target=1)

    Attributes:
        df (pd.DataFrame): a series data frame containing the columns 'question_text' and 'target'
        num_samples (integer): number of samples that will be used to perform a test of hypothesis.
        
    Returns: 
        Two pd.Series of float numbers: the first one contains words that are more likely to appear
        in questions in which the target is zero, the second one contains words that are more likely to appear
        in questions in which the target is one.
"""

    sample_size = df.target.sum()
    sampled_dictionaries=[]
    for i in range(num_samples):
        sampled_questions = df[df.target==0].sample(n=sample_size).question_text
        sampled_dictionaries.append( get_word_count(sampled_questions))
    word_counts = pd.concat(sampled_dictionaries, axis=1, sort=True).fillna(0)
    quantile_05 = word_counts.quantile(q= 0.05, axis=1)
    quantile_95 = word_counts.quantile(q= 0.95, axis=1)
    mean =  word_counts.mean(axis=1)
    std_dev = word_counts.std(axis=1)
    
    insincere_questions = df[df.target==1].question_text
    insincere_word_counts = get_word_count(insincere_questions)

    stats = pd.concat([insincere_word_counts, quantile_05, mean, quantile_95, std_dev], axis=1, sort=True).dropna()
    stats.columns = ['insincere_count', 'quantile_05', 'mean','quantile_95', 'std_dev']
    
    bad_words = stats[stats.insincere_count>stats.quantile_95].copy()
    good_words = stats[stats.insincere_count<stats.quantile_05].copy()

    bad_words = (bad_words.insincere_count - bad_words.quantile_95)/bad_words.std_dev
    good_words = (good_words.quantile_05 - good_words.insincere_count)/good_words.std_dev
    return good_words, bad_words


def get_word_count(string_column):
    """ Given a series of strings, this methods computes, for each word, the 
    ratio number_of_times_the_word_appeared/total_number_of_words after removing
    stop words 
    
    

    Example:
        >> string_column = pd.Series(["wouldn't you like", 'why Ohio?', 'like Ohio'])
        >> get_word_count(string_column)
        like     0.4
        ohio     0.4
        would    0.2
        dtype: float64
        

    Attributes:
        string_column (pd.Series): a series of strings
        
    Returns: 
        A pd.Series of float numbers (the quotient 
        number_of_times_the_word_appeared/total_number_of_words) indexed by each non stop 
        word in string_column.
"""
    stop_words = set(stopwords.words('english'))
    string_series = string_column.str.lower().str.decode('ascii', 'ignore')
    long_string = ' '.join(string_series.values)     
    words = word_tokenize(long_string)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]  
    word_dict= {}
    for w in words:
        word_dict[w]=(word_dict[w]+1) if w in word_dict else 1
    dic_series=pd.Series(word_dict)
    dic_series/=dic_series.sum()
    return dic_series
