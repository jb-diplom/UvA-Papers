# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:05:27 2020

@author: Janice
"""
#%% imports

import math
from gensim import models, corpora
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython import display
from IPython.core.display import HTML
from gensim.matutils import softcossim
import datetime
import time
import pickle
import glob

import gensim.downloader as api
from dateutil.parser import *

import pyLDAvis
import pyLDAvis.gensim
from bokeh.io import  show, output_notebook, output_file

import matplotlib.pyplot as plt
import plotly.express as px

import warnings
# Suppress annoying deprecation messages which I'm not going to fix yet
# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import importlib
# importlib.import_module("rssreader.reader")
importlib.import_module("reader")
from reader import loadAllFeedsFromFile,loadPickleArticles
importlib.import_module("topicmap")
from topicmap import getDocList, getCustomStopWords, deriveTopicMaps, updateDictionaryByFuzzyRelevanceofTopics

# import globals
#%% loadWordEmbeddings
global fasttext_model300   # Defined as global variable due to very long loading times
global w2v_model           # Defined as global variable due to very long loading times
# fasttext_model300=False
# w2v_model=False
# glove-wiki-gigaword-300

def loadWordEmbeddings(modelName="w2v_model"):
    """
    Loads the named word embedding model (used in soft CosineSimilarity test)
    NOTE1:  the first call for "fasttext" includes a download of ca. 1GB of data, 
            which takes ca. 30 minutes, subsequent calls load this from disk 
            which also takes 2-3 minutes
    NOTE2:  the first call for "GloVe (w2v_model)" includes a download of ca. 
            60MB of data, which takes ca. 3 minutes, subsequent calls load this 
            from disk which also takes 30 seconds. Only load if not already bound            
    Parameters
    ----------
    modelName : TYPE, optional
        DESCRIPTION. The word embedding model to load. The default is "w2v_model".
        Other supported model is fasttext_model300

    Returns
    -------
    None. The loaded model ist stored in the global variables fasttext_model300 
    or w2v_model
    """
    global w2v_model
    global fasttext_model300
    if modelName == "w2v_model" and 'w2v_model' not in dir():
        print ("Loading GloVe Embeddings. This may take some time")
        w2v_model = api.load("glove-wiki-gigaword-50")
    elif modelName == "fasttext_model300" and 'fasttext_model300' not in dir():
        print ("Loading Fasttext Embeddings. This will take a very long time")
        fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    else:
        print (modelName, "is unknown")
    return

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

#%% getWordEmbeddingModel TODO try out other word embedding (fasttext_model300)
def getWordEmbeddingModel():
    # Download or load the WordEmbedding models
    # loadWordEmbeddings("w2v_model")
    return api.load("glove-wiki-gigaword-50")

#%% softCosineSimilarityTest
#   https://www.machinelearningplus.com/nlp/cosine-similarity/

def softCosineSimilarityTest(numtestdocs=20):
    # documents=getTestDocuments()
    # documents=getSampleDocs(numtestdocs)
    documents=getDocList(limit=numtestdocs)
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

#%% savePickle

def saveDFPickle(df):
    now=datetime.datetime.now()
    runtimeStr=now.strftime("%d%m%Y_%H%M%S")    # for saving datafiles uniquely
    outfileName="Gensim/runtime_data/softcosmatrix" + runtimeStr + ".pickle"
    with open(outfileName, 'wb') as outfile:
        pickle.dump(df, outfile, pickle.HIGHEST_PROTOCOL)
        
    return outfileName

def loadMatrixFile(path = "../runtime_data" ):
    matrices=[]
    for file in glob.glob(path + "/*.pickle"):
        print ("loading file: ", file)
        matrix=loadPickleArticles(file)
        matrices.append(matrix)
    return matrices

#%% Data Reduction and plotting
# TODO do TruncateSVD and/or TSNE (see James Baker )
# See D:/Janice/github/Capstone/jlbCapstoneClustering.py graphVectorSpace
def calculateXYZByPCAMethod(df, clusterNumber=20, threshold=0.5):
    """ 
    - copy given matrix
    - remove all rows/columns where all values are below given 
    - apply spectral analysis for colouring --> colour
    - apply PCA dimension reduction --> x,y,z
    - add x,y,z and colour and return matrix
    threshold 
    """
    df2=df.copy(deep=True)
    #set all values below threshold to 0
    df2=df2.applymap(lambda x: np.nan if x < threshold or x ==1 else x)
    # compress down to relevant value only 
    df2=df2.dropna(how='all')     # dropb nan rows
    empty_cols = [col for col in df2.columns if df2[col].isnull().all()]
    # Drop these columns from the dataframe
    df2.drop(empty_cols,
            axis=1,
            inplace=True)
    new_df=df2.applymap(lambda x: 0 if pd.isnull(x) else x) # replace nans with 0

    from sklearn.cluster import SpectralClustering
    sc=SpectralClustering(clusterNumber).fit_predict(new_df)

    # determine largest topicality per entry for size of ball in scatterplot
    sizes=[]
    for flt_size in new_df.max(axis=1):
        sizes.append(math.ceil((flt_size-(threshold*0.9))*80))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(new_df)

# Add the resulting coords as additional columns onto the dataframe
    new_df['pca-one'] = pca_result[:,0]
    new_df['pca-two'] = pca_result[:,1]
    new_df['pca-three'] = pca_result[:,2]

# Add the spectral analysis as additional column onto the dataframe
    new_df['specGroup'] = sc
    new_df['size'] = sizes
# TODO add publication date for optionally colouring according to date
# TODO add feedname date for optionally colouring according to feedname
# TODO add article sizedate for optionally sizing balls according to article size

    return new_df

def show3D(df, title=""):

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
    plt.title(title, fontsize=20)
    plt.tight_layout()

    plt.show()
    return

def plotScatter3D(df, title, allDict, notebook=True):
    import plotly
    import plotly.graph_objs as go
    statTooltips=[]
    for key in df.index:
        try:
            statTooltips.append(tooltipText(allDict[key]))
        except:
            print (key, "not found")

    trace = go.Scatter3d(
        x=df['pca-one'],
        y=df['pca-two'],
        z=df['pca-three'],
        mode='markers',
        marker=dict(
            size=df["size"],
            color=df["specGroup"],
            colorscale='Viridis',
            # symbol=df["specGroup"], # TODO actually want Feedname
            showscale=True,
            opacity=0.6
        ),
        text=statTooltips,
        hoverinfo='text',
    )
    # Configure the layout.
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    # plot_figure.update_layout(title:title)
    # plt.title(title, fontsize=16)
    # plt.tight_layout()

    # Render the plot.
    pl=plotly.offline.iplot(plot_figure)
    pl
    if not notebook:
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
    # TODO: see if we can get ngrams into pyLDAvis
    if not notebook:
        output_file("pyDAVis.html")
    else:
        pyLDAvis.enable_notebook()
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
def test3DPlotOfCosineSimilarity(allDict, num=None, matrix=None,
                                 numTopics=40, threshold=0.7, notebook=True):
    if not matrix is not None :
        matrix=deriveSoftCosineSimilarityMatrix(allDict, num)
    matout=calculateXYZByPCAMethod(matrix,numTopics,threshold)
    plotScatter3D(matout,"10 Groups, threshold 0.69", allDict,notebook=notebook)

    # can also do a simple perspective plot with matplot
    # show3D(matout)

    # In case you need to clean up a previously used and abused matrix ..
    # newmatr=newmatr_svae.drop(columns=['pca-one', 'pca-two','pca-three','specGroup'])

    return
#%% Test method for displaying 3d plot with topic
def testCase():
    """
    Get docs from file, get list of titles+content, calculate (40) topics using 3,4 ngrams
    map topics Add list of topics to each entry of the given allEntryDict for each topic
    that has an LDA fuzzy relevance (see fuzzywuzzy process) of greater than the
    specified threshold. Calculate SoftCosine-Similarity matrix with WordEmbeddings
    fasttext_model300 (dimension 300) or GloVe (dimension 50)
    save matrix to file
    do spectral analysis and dimension reduction (PCA method) on similarity matrix
    plotScatter3D with tool tips
    """
    allDict=loadAllFeedsFromFile()
    docl=getDocList(allDict, reloaddocs=False,stop_list=getCustomStopWords())
    topics= deriveTopicMaps(docl, maxNum=40, ngram_range=(3,4))
    updateDictionaryByFuzzyRelevanceofTopics(topics,allDict, limit=None, threshold=60,remove=True )
    trix=deriveSoftCosineSimilarityMatrix(allDict)
    saveDFPickle(trix)
    test3DPlotOfCosineSimilarity(allDict,None,trix)
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

# m=loadMatrixFile()
# test3DPlotOfCosineSimilarity(allDict, 100, m[0])

# m=loadMatrixFile()
# allDict=loadAllFeedsFromFile()
# test3DPlotOfCosineSimilarity(allDict, 500, m[1], 6, 0.5, notebook=False)


