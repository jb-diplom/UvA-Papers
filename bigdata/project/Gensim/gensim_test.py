# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:05:27 2020

@author: Janice
"""
#%% imports

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
from dateutil.parser import *

import pyLDAvis
import pyLDAvis.gensim
from bokeh.io import  show, output_notebook, output_file

import matplotlib.pyplot as plt

import warnings
# Suppress annoying deprecation messages which I'm not going to fix yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

import importlib
# importlib.import_module("rssreader.reader")
importlib.import_module("reader")
from reader import getSampleDocs
importlib.import_module("topicmap")
from topicmap import getDocList, getCustomStopWords

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



#%% tfidfTest TODO probably delete since it's a test method
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
        
#%% getTestDocuments TODO Delete
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
    
#%% renderTable
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

#%% cosineSimilarityTest TODO delete
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

#%% getWordEmbeddingModel TODO try out other word embedding (fasttext_model300)
def getWordEmbeddingModel():
    # Download or load the WordEmbedding models
        
    return w2v_model

#%% softCosineSimilarityTest
#   https://www.machinelearningplus.com/nlp/cosine-similarity/

def softCosineSimilarityTest(numtestdocs=20):
    # documents=getTestDocuments()
    documents=getSampleDocs(numtestdocs) # TODO replace with getDocList
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

#%% deriveSoftCosineSimilarityMatrix
#   https://www.machinelearningplus.com/nlp/cosine-similarity/

def deriveSoftCosineSimilarityMatrix(allDict, limit=None):
    # documents=getTestDocuments()
    docsZip=getDocList(allDict,limit,stop_list=getCustomStopWords(), with_ids=True)

    documents=[]
    ids=[]
    for i,j in docsZip:
        documents.append(j)
        ids.append(i)
    model=getWordEmbeddingModel()
    # Create gensim Dictionary of unique IDs of all words in all documents
    # pyDAVis param "d"
    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

    # Prepare the similarity matrix
    # TODO Check if some of these parameters can be used to begin with rather than filtering later
    # TODO Shouldn't the tf_idf from below be put into this call?
    similarity_matrix = model.similarity_matrix(    dictionary, 
                                                    tfidf=None, 
                                                    threshold=0.0, 
                                                    exponent=2.0, 
                                                    nonzero_limit=100)
    
    # Convert the sentences into bag-of-words vectors.
    sentences=[]     # pyDAVis param "c"
    for doc in documents:
        sentences.append(dictionary.doc2bow(simple_preprocess(doc)))

    # Create a TF-IDF model. TF-IDF encoding represents words as their 
    # relative importance to the whole document in a collection of documents,
    # i.e. the sentences.
    # pyDAVis param "lda"
    tf_idf = models.TfidfModel(sentences)

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

#%% Data Reduction and plotting
# TODO do TruncateSVD and/or TSNE (see James Baker )
# See D:/Janice/github/Capstone/jlbCapstoneClustering.py graphVectorSpace
def calculateXYZByPCAMethod(df, clusterNumber=20, threshold=0.5):

    try:   # remove any previous calculated values
        df.drop(columns=['pca-one', 'pca-two','pca-three'])
        df.drop(columns=['specGroup'])
    except:
        print("No columns to drop")

    df2=df.copy()
    #set all values below threshold to 0
    df2=df2.applymap(lambda x: np.nan if x < threshold or x ==1 else x)
    # compress down to relevant value only 
    df2.index.dropna(how='all')     # dropb nan rows
    empty_cols = [col for col in df2.columns if df2[col].isnull().all()]
    # Drop these columns from the dataframe
    df2.drop(empty_cols,
            axis=1,
            inplace=True)
    new_df=df2.applymap(lambda x: 0 if pd.isnull(x) else x) # replace nans with 0

    from sklearn.cluster import SpectralClustering
    sc=SpectralClustering(clusterNumber).fit_predict(new_df)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(new_df)

# Add the resulting coords as additional columns onto the dataframe
    new_df['pca-one'] = pca_result[:,0]
    new_df['pca-two'] = pca_result[:,1]
    new_df['pca-three'] = pca_result[:,2]

# Add the spectral analysis as additional column onto the dataframe
    new_df['specGroup'] = sc
# TODO add publication date for optionally colouring according to date
# TODO add feedname date for optionally colouring according to feedname
# TODO add article sizedate for optionally sizing balls according to article size

    return new_df

def show3D(df):

    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df.loc[:]["pca-one"],
        ys=df.loc[:]["pca-two"],
        zs=df.loc[:]["pca-three"],
        c=df.loc[:]["specGroup"],
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')

    # hover=plt.select(dict(type=HoverTool))
    # ax.legend.click_policy="hide"
    # hover.tooltips={"id": "@index", "publication": "@pca-one", "content":"@pca-two", "category":"@specGroup"}

    plt.show()
    return

def plotScatter3D(df, title, allDict):
    import plotly
    import plotly.graph_objs as go
    statTooltips=[]
    for key in df.index:
        statTooltips.append(tooltipText(allDict[key]))

    trace = go.Scatter3d(
        x=df['pca-one'],
        y=df['pca-two'],
        z=df['pca-three'],
        mode='markers',
        marker=dict(
            size=0,
            color=df["specGroup"],
            colorscale='Jet',
            showscale=True,
            opacity=0.5
        ),
        text=statTooltips,
        hoverinfo='text',
    )
    # Configure the layout.
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    
    # Render the plot.
    pl=plotly.offline.iplot(plot_figure)
    pl
    plotly.offline.plot(plot_figure, filename='file.html')
    return trace

#%% tooltipText
def tooltipText(rssEntry):
    """
    Pseudo HTML string for displaying the RSS-entry infos, currently 
    title, feedname, date of publication and author

    Parameters
    ----------
    rssEntry : TYPE
        DESCRIPTION.

    Returns
    -------
    str : TYPE
        DESCRIPTION.

    """
    src = rssEntry["feed_name"]
    auth = getAuthorFromRssEntry(rssEntry)
    if hasattr(rssEntry , "updated"):
        dt=parse(rssEntry.updated, ignoretz=True)
    else:
        dt=parse(rssEntry.published, ignoretz=True)
    title = rssEntry.title

    datestr=dt.strftime("%d/%m/%Y, %H:%M:%S")
    items=[]
    items.append(f"Title:           {title : <10}")
    items.append(f"Feed:         {src: <10}")
    items.append(f"Published: {datestr: <10}")
    items.append(f"Author:       {auth : <10}")

    str=""
    for token in items:
        str=str+(token)+("<br>")

    return str

#%% Authors over all Articles
def getAuthorFromRssEntry(val):
    """
    Get the author(s) string for the given RSS-entry
    Parameters
    ----------
    val : dict and RSS-entry
        DESCRIPTION.
    Returns
    -------
    str just the name or names of the authors or empty string if none
        DESCRIPTION.

    """
    authors=[]
    nwith=0
    nwithout=0
    try:
        # if hasattr(val , "authors"):
        #     authors.extend([n['name'] for n in val.authors])
        if hasattr(val , "author"):
            if ',' in val.author:
                authors.extend(val.author.split (",") for val in authors)
            elif ' and ' in val.author:
                authors.extend(val.author.split (" and "))
            else:
                authors.append(val.author)
            nwith +=1
        else:
            nwithout +=1
    except:
        nwithout +=1

    for ix, auth in enumerate(authors):
        if ' and ' in auth:
            authors.extend(auth.split (" and "))
     # for auth in authors:
     #     if (len(auth))

    return " ".join(authors)

#%% preparePyLDAvisData
#   https://www.machinelearningplus.com/nlp/cosine-similarity/

def preparePyLDAvisData(allDict, limit=None, numTopics=30):

    docsZip=getDocList(allDict,limit,stop_list=getCustomStopWords(), with_ids=True)
    documents=[]
    ids=[]
    for i,j in docsZip:
        documents.append(j)
        ids.append(i)
    model=getWordEmbeddingModel()
    # Create gensim Dictionary of unique IDs of all words in all documents
    # pyDAVis param "d"
    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

    # Prepare the similarity matrix
    # TODO Check if some of these parameters can be used to begin with rather than filtering later
    similarity_matrix = model.similarity_matrix(    dictionary, 
                                                    tfidf=None, 
                                                    threshold=0.0, 
                                                    exponent=2.0, 
                                                    nonzero_limit=100)
    
    # Convert the sentences into bag-of-words vectors.
    sentences=[]     # pyDAVis param "c"
    for doc in documents:
        sentences.append(dictionary.doc2bow(simple_preprocess(doc)))

    ldamodel = models.ldamodel.LdaModel(sentences, num_topics=numTopics, id2word = dictionary, passes=50)

    # # create 1xN vector filled with 1,2,..N
    # len_array = np.arange(len(sentences)) 
    # # create NxN array filled with 1..N down, 1..N across
    # xx, yy = np.meshgrid(len_array, len_array)
    # # Iterate over the 2d matrix calculating
    # theMatrix=[[round(softcossim(sentences[i],sentences[j], similarity_matrix) ,2) 
    #             for i, j in zip(x,y)] 
    #             for y, x in zip(xx, yy)]
    
    # cossim_mat = pd.DataFrame(theMatrix, index=ids, columns=ids)

    return (ldamodel, sentences, dictionary)

#%% showPyLDAvis

def showPyLDAvis(allDict, notebook=True, numTopics=30):
    if not notebook:
        output_file("pyDAVis.html")
    dataTuple=preparePyLDAvisData(allDict, limit=None, numTopics=numTopics)
    data = pyLDAvis.gensim.prepare(dataTuple[0],dataTuple[1],dataTuple[2])
    if notebook:
        p=pyLDAvis.display(data)
    else:
        p=pyLDAvis.show(data) # displays in own window combined with output_file
    show(p)
    return

#%% format_vertical_headers
def format_vertical_headers(df):
    """Display a dataframe with vertical column headers"""
    styles = [dict(selector="th", props=[('width', '40px')]),
              dict(selector="th.col_heading",
                   props=[("writing-mode", "vertical-rl"),
                          ('transform', 'rotateZ(180deg)'), 
                          ('height', '290px'),
                          ('vertical-align', 'top')])]
    return (df.fillna('').style.set_table_styles(styles))

#%% testPerfOfsoftCosineSimilarity
def testPerfOfsoftCosineSimilarity(numdocs=20):
    tic = time.perf_counter()
    mat=softCosineSimilarityTest(numdocs)
    toc = time.perf_counter()
    print(f"Cosine Similarity for {numdocs} docs performed in {toc - tic:0.4f} seconds")
    mat.to_csv('outdata/' + str(numdocs) + 'x' + str(numdocs) + '.csv', sep='\t')
    return mat


#%% Test 3d Plotting od cosine similarity matrix
def test3DPlotOfCosineSimilarity(allDict, num=None):

    matrix=deriveSoftCosineSimilarityMatrix(allDict, num)
    matout=calculateXYZByPCAMethod(matrix,8,0.69)
    plotScatter3D(matout,"10 Groups, threshold 0.69", allDict)

    # can also do a simple perspective plot with matplot
    show3D(matout)

    # In case you need to clean up a previously used and abused matrix ..
    # newmatr=newmatr_svae.drop(columns=['pca-one', 'pca-two','pca-three','specGroup'])

    return
#%% Test run methods

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
# showPyLDAvis(allDict, False)
# showPyLDAvis(smallDict(allDict,10), False, 30)
# test3DPlotOfCosineSimilarity(allDict, 500)