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
        ## alternative line: string_column.str.lower().str.decode('ascii', 'ignore')
        #str_series = X[self.colname].apply(str).apply(lambda x: ''.join([i if ord(i) < 128 else ' ' for i in x]))
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
    def __init__(self, col1=None, col2=None, model=None, norm_model=None):
        self.col1 = col1
        self.col2 = col2
        self.model = model
        self.norm_model = norm_model

    def transform(self, X):        
        suffix = "_" + self.col1 + "_"+ self.col2

        strings = pd.DataFrame(index = X.index)
        ## alternative line: string_column.str.lower().str.decode('ascii', 'ignore')
        #strings['1'] = X[self.col1].apply(str).apply(lambda x: ''.join([i if ord(i) < 128 else ' ' for i in x]))
        #strings['2'] = X[self.col2].apply(str).apply(lambda x: ''.join([i if ord(i) < 128 else ' ' for i in x]))

        
        strings['1'] = X[self.col1].str.decode('ascii', 'ignore')
        strings['2'] = X[self.col2].str.decode('ascii', 'ignore')
        embeddings = pd.DataFrame(index = X.index)
        embeddings['1'] = strings['1'].apply(lambda x: sent2vec(x,self.model)).apply(np.nan_to_num)
        embeddings['2'] = strings['2'].apply(lambda x: sent2vec(x,self.model)).apply(np.nan_to_num)
        
        self.debbug = embeddings.copy()
        
        df = pd.DataFrame(index = X.index)
        df['wmd' + suffix] = strings.apply(lambda x: wmd(x['1'], x['2'], self.model), axis=1) 
        df['norm_wmd' + suffix] = strings.apply(lambda x: wmd(x['1'], x['2'], self.norm_model), axis=1)  
        df['common_words' + suffix] = strings.apply(
            lambda x: len(set(x['1'].lower().split()).intersection(set(x['2'].lower().split()))), axis=1)
        df['fuzz_qratio' + suffix] = strings.apply(lambda x: fuzz.QRatio(x['1'], x['2']), axis=1)
        df['fuzz_WRatio' + suffix] = strings.apply(lambda x: fuzz.WRatio(x['1'], x['2']), axis=1)
        df['fuzz_partial_ratio' + suffix] = strings.apply(lambda x: fuzz.partial_ratio(x['1'], x['2']), axis=1)
        df['fuzz_partial_token_set_ratio' + suffix] = strings.apply(
            lambda x: fuzz.partial_token_set_ratio(x['1'], x['2']), axis=1)
        df['fuzz_partial_token_sort_ratio' + suffix] = strings.apply(
            lambda x: fuzz.partial_token_sort_ratio(x['1'], x['2']), axis=1)
        df['fuzz_token_set_ratio' + suffix] = strings.apply(lambda x: fuzz.token_set_ratio(x['1'], x['2']), axis=1)
        df['fuzz_token_sort_ratio' + suffix] = strings.apply(lambda x: fuzz.token_sort_ratio(x['1'], x['2']), axis=1)
        df['cosine_distance'  + suffix] = embeddings.apply(lambda x: cosine(x['1'], x['2']),axis=1)
        df['cityblock_distance' + suffix] = embeddings.apply(lambda x: cityblock(x['1'], x['2']),axis=1)
        df['jaccard_distance' + suffix] = embeddings.apply(lambda x: jaccard(x['1'], x['2']),axis=1)
        df['canberra_distance' + suffix] = embeddings.apply(lambda x: canberra(x['1'], x['2']),axis=1)
        df['euclidean_distance' + suffix] = embeddings.apply(lambda x: euclidean(x['1'], x['2']),axis=1)
        df['minkowski_distance' + suffix] = embeddings.apply(lambda x: minkowski(x['1'], x['2']),axis=1)
        df['braycurtis_distance' + suffix] = embeddings.apply(lambda x: braycurtis(x['1'], x['2']),axis=1)
        return df


    
    
    
    
    
    
