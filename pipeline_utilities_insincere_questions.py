import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin  
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
stop_words = set(stopwords.words('english'))



def wmd(s1, s2, model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

def sent2vec(s, model):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


class stringTransformer(TransformerMixin):
    """ T.
        
    Example:   
        >> 
        >> 
        >> 


    Attributes:
       at1:  

       at2:
    """
    def __init__(self, colname=None, model=None):
        self.colname = colname
        self.model = model

    def transform(self, X):
        suffix = "_" + self.colname 
        str_series = X[self.colname].str.decode('ascii', 'ignore')
        embedding = str_series.apply(lambda x: sent2vec(x,self.model)).apply(np.nan_to_num)
        
        df = pd.DataFrame(index=str_series.index)          
        df['len' + suffix] = str_series.apply(lambda x: len(x))
        df['len_char' + suffix] = str_series.apply(lambda x: len(''.join(set(x.replace(' ', '')))))
        df['len_word' + suffix] = str_series.apply(lambda x: len(x.split()))
        
        
        df['skew' + suffix] = embedding.apply(lambda x: skew(x)) 
        df['kur' + suffix] = embedding.apply(lambda x: kurtosis(x))  
        return df

    

class stringComparison(TransformerMixin):
    """ T.
        
    Example:   
        >> 
        >> 
        >> 


    Attributes:
       at1:  

       at2:
    """
    def __init__(self, col=None, sentence=None, suffix=None, model=None, norm_model=None):
        self.col = col
        self.sentence = sentence
        self.suffix=suffix
        self.model = model
        self.norm_model = norm_model

    def transform(self, X):   

        
        strings = X[self.col].str.decode('ascii', 'ignore')
        embedded_strings = strings.apply(lambda x: sent2vec(x,self.model)).apply(np.nan_to_num)

        self.debbug = embedded_strings
        self.debbug2 = embedded_sentence
        
        df = pd.DataFrame(index = X.index)
        
        ## for sentence in sentence_list:
        
        sentence=self.sentence
        embedded_sentence = np.nan_to_num(sent2vec(sentence,self.model))
        
        
        suffix = self.suffix
        df['wmd' + suffix] = strings.apply(lambda x: wmd(x, sentence, self.model)) 
        df['norm_wmd' + suffix] = strings.apply(lambda x: wmd(x, sentence, self.norm_model))  
        df['common_words' + suffix] = strings.apply(
            lambda x: len(set(x.lower().split()).intersection(set(sentence.lower().split()))))
        df['fuzz_qratio' + suffix] = strings.apply(lambda x: fuzz.QRatio(x, sentence))
        df['fuzz_WRatio' + suffix] = strings.apply(lambda x: fuzz.WRatio(x, sentence))
        df['fuzz_partial_ratio' + suffix] = strings.apply(lambda x: fuzz.partial_ratio(x, sentence))
        df['fuzz_partial_token_set_ratio' + suffix] = strings.apply(
            lambda x: fuzz.partial_token_set_ratio(x, sentence))
        df['fuzz_partial_token_sort_ratio' + suffix] = strings.apply(
            lambda x: fuzz.partial_token_sort_ratio(x, sentence))
        df['fuzz_token_set_ratio' + suffix] = strings.apply(lambda x: fuzz.token_set_ratio(x, sentence))
        df['fuzz_token_sort_ratio' + suffix] = strings.apply(lambda x: fuzz.token_sort_ratio(x, sentence))
        
        df['cosine_distance'  + suffix] = embedded_strings.apply(lambda x: cosine(x, embedded_sentence))
        df['cityblock_distance' + suffix] = embedded_strings.apply(lambda x: cityblock(x, embedded_sentence))
        df['jaccard_distance' + suffix] = embedded_strings.apply(lambda x: jaccard(x, embedded_sentence))
        df['canberra_distance' + suffix] = embedded_strings.apply(lambda x: canberra(x, embedded_sentence))
        df['euclidean_distance' + suffix] = embedded_strings.apply(lambda x: euclidean(x, embedded_sentence))
        df['minkowski_distance' + suffix] = embedded_strings.apply(lambda x: minkowski(x, embedded_sentence))
        df['braycurtis_distance' + suffix] = embedded_strings.apply(lambda x: braycurtis(x, embedded_sentence))
        return df


    
    
    
    
    
    
