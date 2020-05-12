# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:05:27 2020

@author: Janice
"""
#%%


from gensim import models, corpora
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython import display
from gensim.matutils import softcossim 

import gensim.downloader as api
import warnings
# Suppress annoying deprecation messages which I'm not going to fix yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

import importlib
# importlib.import_module("rssreader.reader")
importlib.import_module("reader")
from reader import getSampleDocs,getDocList

import time


"""
NOTE:   the first call for "fasttext" includes a download of ca. 1GB of data, 
        which takes ca. 30 minutes, subsequent calls load this from disk 
        which also takes 2-3 minutes
"""
# if 'fasttext_model300' not in dir():
#     print ("Loading Fasttext Embeddings. This will take a very long time")
#     fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
"""
NOTE:   the first call for "GloVe" includes a download of ca. 60MB of data, 
        which takes ca. 3 minutes, subsequent calls load this from disk 
        which also takes 30 seconds. Only load if not already bound
"""
if 'w2v_model' not in dir():
    print ("Loading GloVe Embeddings. This may take some time")
    w2v_model = api.load("glove-wiki-gigaword-50")



#%%
def tfidfTest():

    
    documents = ["This is the first line",
                 "This is the second sentence",
                 "This third document"]
    
    # Create the Dictionary and Corpus
    mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]
    
    # Show the Word Weights in Corpus
    for doc in corpus:
        print([[mydict[id], freq] for id, freq in doc])
    
    # [['first', 1], ['is', 1], ['line', 1], ['the', 1], ['this', 1]]
    # [['is', 1], ['the', 1], ['this', 1], ['second', 1], ['sentence', 1]]
    # [['this', 1], ['document', 1], ['third', 1]]
    
    # Create the TF-IDF model
        
    tfidf = models.TfidfModel(corpus, smartirs='ntc')
    
    # Show the TF-IDF weights
    for doc in tfidf[corpus]:
        print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])
        
#%%
def getTestDocuments():
        # Define the documents
        doc_trump = "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin"
        doc_election = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election"
        doc_putin = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career"
        doc_soup = "Soup is a primarily liquid food, generally served warm or hot (but may be cool or cold), that is made by combining ingredients of meat or vegetables with stock, juice, water, or another liquid. "
        doc_noodles = "Noodles are a staple food in many cultures. They are made from unleavened dough which is stretched, extruded, or rolled flat and cut into one of a variety of shapes."
        doc_dosa = "Dosa is a type of pancake from the Indian subcontinent, made from a fermented batter. It is somewhat similar to a crepe in appearance. Its main ingredients are rice and black gram."
        documents = [doc_trump, doc_election, doc_putin, doc_soup, doc_noodles, doc_dosa]
        return documents
    
#%%
def renderTable(df1):
        # create output widgets
    widget1 = widgets.Output()
    
    # render in output widgets
    with widget1:
        display.display(df1)
    
    # create HBox
    hbox = widgets.HBox([widget1])
    
    # render hbox
    hbox

#%%
def cosineSimilarityTest():

    documents = getTestDocuments()
    # Create the Document Term Matrix
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)
    
    # OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, 
                      columns=count_vectorizer.get_feature_names(), 
                      index=['doc_trump', 'doc_election', 'doc_putin'])
    print(cosine_similarity(df, df))
    """
    [[1.         0.51480485 0.38890873]
     [0.51480485 1.         0.38829014]
     [0.38890873 0.38829014 1.        ]]
    
    Interpretation: 
    doc_trump is more similar to doc_election (0.51) than to doc_putin (0.39)
    """
    return df

#%%
def getWordEmbeddingModel():
    # Download or load the WordEmbedding models
        
    return w2v_model

#%%
#   https://www.machinelearningplus.com/nlp/cosine-similarity/

def softCosineSimilarityTest(numtestdocs=20):
    # documents=getTestDocuments()
    documents=getSampleDocs(numtestdocs)
    model=getWordEmbeddingModel()
    # Create gensim Dictionary of unique IDs of all words in all documents
    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

    # Prepare the similarity matrix
    similarity_matrix = model.similarity_matrix(    dictionary, 
                                                    tfidf=None, 
                                                    threshold=0.0, 
                                                    exponent=2.0, 
                                                    nonzero_limit=100)
    
    # Convert the sentences into bag-of-words vectors.
    sentences=[]
    for doc in documents:
        sentences.append(dictionary.doc2bow(simple_preprocess(doc)))
        
    # Create a TF-IDF model. TF-IDF encoding represents words as their 
    # relative importance to the whole document in a collection of documents,
    # i.e. the sentences.
    # tf_idf = models.TfidfModel(sentences)
    # print("tf_idf:", tf_idf)
    
    # create 1xN vector filled with 1,2,..N    
    len_array = np.arange(len(sentences)) 
    # create NxN array filled with 1..N down, 1..N across
    xx, yy = np.meshgrid(len_array, len_array)
    # Iterate over the 2d matrix calculating
    theMatrix=[[round(softcossim(sentences[i],sentences[j], similarity_matrix) ,2) 
                for i, j in zip(x,y)] 
                for y, x in zip(xx, yy)]
    
    names=[]        # for identifying rows and columns
    jj=0
    for doc in documents:
        names.append(str(jj) + " " + doc[:15] + "\t")
        jj +=1
        
    cossim_mat = pd.DataFrame(theMatrix, index=names, columns=names)

    return cossim_mat

#%%
#   https://www.machinelearningplus.com/nlp/cosine-similarity/

def deriveSoftCosineSimilarityMatrix(allDict, limit=None):
    # documents=getTestDocuments()
    docsZip=getDocList(allDict,limit,with_ids=True)
    documents=[]
    ids=[]
    for i,j in docsZip:
        documents.append(j)
        ids.append(i)
    model=getWordEmbeddingModel()
    # Create gensim Dictionary of unique IDs of all words in all documents
    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

    # Prepare the similarity matrix
    similarity_matrix = model.similarity_matrix(    dictionary, 
                                                    tfidf=None, 
                                                    threshold=0.0, 
                                                    exponent=2.0, 
                                                    nonzero_limit=100)
    
    # Convert the sentences into bag-of-words vectors.
    sentences=[]
    for doc in documents:
        sentences.append(dictionary.doc2bow(simple_preprocess(doc)))
        
    # create 1xN vector filled with 1,2,..N    
    len_array = np.arange(len(sentences)) 
    # create NxN array filled with 1..N down, 1..N across
    xx, yy = np.meshgrid(len_array, len_array)
    # Iterate over the 2d matrix calculating
    theMatrix=[[round(softcossim(sentences[i],sentences[j], similarity_matrix) ,2) 
                for i, j in zip(x,y)] 
                for y, x in zip(xx, yy)]
    
    cossim_mat = pd.DataFrame(theMatrix, index=ids, columns=ids)

    return cossim_mat

#%%
def format_vertical_headers(df):
    """Display a dataframe with vertical column headers"""
    styles = [dict(selector="th", props=[('width', '40px')]),
              dict(selector="th.col_heading",
                   props=[("writing-mode", "vertical-rl"),
                          ('transform', 'rotateZ(180deg)'), 
                          ('height', '290px'),
                          ('vertical-align', 'top')])]
    return (df.fillna('').style.set_table_styles(styles))

#%%
def testPerfOfsoftCosineSimilarity(numdocs=20):
    tic = time.perf_counter()
    mat=softCosineSimilarityTest(numdocs)
    toc = time.perf_counter()
    print(f"Cosine Similarity for {numdocs} docs performed in {toc - tic:0.4f} seconds")
    mat.to_csv('outdata/' + str(numdocs) + 'x' + str(numdocs) + '.csv', sep='\t')


#%%

# Do one time only to get wordnet for Lemmatization

# testPerfOfsoftCosineSimilarity(10)
# soft_cosine_similarity_matrix(sentences)
# similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)
# testPerfOfsoftCosineSimilarity(5)
# testPerfOfsoftCosineSimilarity(100)
# testPerfOfsoftCosineSimilarity(1000)
# testPerfOfsoftCosineSimilarity(2000)
# testPerfOfsoftCosineSimilarity(3000)
# matr=deriveSoftCosineSimilarityMatrix(allDict, 10)
