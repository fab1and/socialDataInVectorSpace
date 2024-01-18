######################################
### Customer 2 Vec Implementations ###
###     Based on Top2Vec           ###
######################################
# contains functions from: https://github.com/ddangelov/Top2Vec/blob/d625b507aa18a921b7d0a3710d1a4c176f9b8f84/top2vec/Top2Vec.py#L813

def tokenizer(document):

    """

    :param document:  str
    :return: tokenized str
    """

    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import strip_tags

    return simple_preprocess(strip_tags(document), deacc=True)

def return_doc(doc):
    return doc

def _create_topic_vectors(cluster_labels, document_vectors):

    """
    Aggregates C_t in order to create the topic vectors

    :param cluster_labels: cluster lables from HDBSCAN output
    :param document_vectors: array containing document embeddings
    :return: aggregated document vectors by mean
    """

    from sklearn.preprocessing import normalize
    import numpy as np

    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    topic_vectors = normalize(
        np.vstack([document_vectors[np.where(cluster_labels == label)[0]]
                  .mean(axis=0) for label in unique_labels]))
    return topic_vectors

def _find_topic_words_and_scores(topic_vectors, word_vectors, vocab):

    """
    Calculates the cosine similarities of each topic towards the word vectors

    :param topic_vectors:
    :param word_vectors:
    :param vocab:
    :return: topic words w* and topic scores which are
    """

    import numpy as np
    topic_words = []
    topic_word_scores = []

    res = np.inner(topic_vectors, word_vectors) # since normalized inner = cosine_sim
    top_words = np.flip(np.argsort(res, axis=1), axis=1)
    top_scores = np.flip(np.sort(res, axis=1), axis=1)

    for words, scores in zip(top_words, top_scores):
        topic_words.append([vocab[i] for i in words[0:50]])
        topic_word_scores.append(scores[0:50])

    topic_words = np.array(topic_words)
    topic_word_scores = np.array(topic_word_scores)

    return topic_words, topic_word_scores

def _calculate_documents_topic(topic_vectors, document_vectors, dist=True, num_topics=None):

    import numpy as np

    batch_size = 10000
    doc_top = []
    if dist:
        doc_dist = []

    if document_vectors.shape[0] > batch_size:
        current = 0
        batches = int(document_vectors.shape[0] / batch_size)
        extra = document_vectors.shape[0] % batch_size

        for ind in range(0, batches):
            res = np.inner(document_vectors[current:current + batch_size], topic_vectors)

            if num_topics is None:
                doc_top.extend(np.argmax(res, axis=1))
                if dist:
                    doc_dist.extend(np.max(res, axis=1))
            else:
                doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                if dist:
                    doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

            current += batch_size

        if extra > 0:
            res = np.inner(document_vectors[current:current + extra], topic_vectors)

            if num_topics is None:
                doc_top.extend(np.argmax(res, axis=1))
                if dist:
                    doc_dist.extend(np.max(res, axis=1))
            else:
                doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                if dist:
                    doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])
        if dist:
            doc_dist = np.array(doc_dist)
    else:
        res = np.inner(document_vectors, topic_vectors)

        if num_topics is None:
            doc_top = np.argmax(res, axis=1)
            if dist:
                doc_dist = np.max(res, axis=1)
        else:
            doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
            if dist:
                doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

    if num_topics is not None:
        doc_top = np.array(doc_top)
        if dist:
            doc_dist = np.array(doc_dist)

    if dist:
        return doc_top, doc_dist
    else:
        return doc_top

def _reorder_topics(topic_vectors, topic_words, topic_word_scores, doc_top, topic_sizes):

    import numpy as np
    topic_vectors = topic_vectors[topic_sizes.index]
    topic_words = topic_words[topic_sizes.index]
    topic_word_scores = topic_word_scores[topic_sizes.index]
    old2new = dict(zip(topic_sizes.index, range(topic_sizes.index.shape[0])))
    doc_top = np.array([old2new[i] for i in doc_top])
    topic_sizes.reset_index(drop=True, inplace=True)

# assign documents to topic
def _assign_documents_to_topic(document_vectors, topic_vectors, topic_sizes, doc_top, doc_dist, topic_words, topic_word_scores):

    import numpy as np
    import pandas as pd

    doc_top_new, doc_dist_new = _calculate_documents_topic(topic_vectors, document_vectors, dist=True)
    doc_top = np.array(list(doc_top) + list(doc_top_new))
    doc_dist = np.array(list(doc_dist) + list(doc_dist_new))

    topic_sizes_new = pd.Series(doc_top_new).value_counts()
    for top in topic_sizes_new.index.tolist():
        topic_sizes[top] += topic_sizes_new[top]
    topic_sizes.sort_values(ascending=False, inplace=True)
    _reorder_topics(topic_vectors, topic_words, topic_word_scores, doc_top, topic_sizes)

    return doc_top, doc_dist, topic_sizes

def _validate_keywords(keywords, keywords_neg,vocab):

    import numpy as np

    if not (isinstance(keywords, list) or isinstance(keywords, np.ndarray)):
        raise ValueError("keywords must be a list of strings.")

    if not (isinstance(keywords_neg, list) or isinstance(keywords_neg, np.ndarray)):
        raise ValueError("keywords_neg must be a list of strings.")

    keywords_lower = [keyword.lower() for keyword in keywords]
    keywords_neg_lower = [keyword.lower() for keyword in keywords_neg]

    vocab = vocab
    for word in keywords_lower + keywords_neg_lower:
        if word not in vocab:
            raise ValueError(f"'{word}' has not been learned by the model so it cannot be searched.")

    return keywords_lower, keywords_neg_lower

def search_documents_by_keywords(model, keywords,documents, documents_ind, num_docs, word_vectors, word_indexes, vocab,keywords_neg=None, return_documents=True):

    import numpy as np

    _words2word_vectors = word_vectors[[word_indexes[word] for word in keywords]]

    if keywords_neg is None:
        keywords_neg = []

    #keywords, keywords_neg = _validate_keywords(keywords, keywords_neg, vocab)
    word_vecs = _words2word_vectors#(keywords)

    sim_docs = model.docvecs.most_similar(positive=word_vecs,
                                               negative=keywords_neg,
                                               topn=num_docs)


    doc_indexes = [doc[0] for doc in sim_docs if doc[0] in documents_ind]
    doc_scores = np.array([doc[1] for doc in sim_docs if doc[0] in documents_ind])

    doc_ids = np.array(range(0, len(documents)))

    #ToDo: implement threshold regarding score
    if documents is not None and return_documents:
        documents = documents[doc_indexes]
        return documents[:num_docs], doc_scores[:num_docs], doc_ids[:num_docs]
    else:
        return doc_scores, doc_ids

def top2vecReBuild(documents, epochs,min_count,vector_size,use_phrases = True):

    """
    This is an outsources implementation of Angelovs Top2Vec implementation, taken from GitHub.

    :param documents: Comments
    :param epochs: Number of epochs used within Doc2Vec
    :param min_count: minimum word frequency used within Doc2Vec
    :param vector_size: vectors ize used within Doc2Vec
    :param use_phrases: the original implementation also allows to introduce
    :return: document vectors, word vectors, the vocabulary and the Doc2Vec instance
    """

    import tempfile
    import numpy as np
    from tqdm.notebook import tqdm
    tqdm.pandas()
    from sklearn.preprocessing import normalize
    from Top2VecModule import tokenizer
    from Top2VecModule import return_doc
    from sklearn.feature_extraction.text import CountVectorizer

    print('tokenizing corpus')

    # PreProcessing of Top2Vec via gensim
    tokenized_corpus = [tokenizer(doc) for doc in documents]


    # initialize doc2vec model
    doc2vec_args = {"vector_size": vector_size,
                    "min_count": min_count, # number of times the word appears in the data
                    # set to 1 for testing purposes
                    "window": 7, #ToDo: tune window_size (median)
                    "sample": 1e-5,
                    "negative": 0,
                    "hs": 1,
                    "epochs": epochs,
                    "dm": 0,
                    "dbow_words": 1}

    # PreProcess
    processed = [' '.join(tokenizer(doc)) for doc in documents]
    lines = '\n'.join(processed)
    temp = tempfile.NamedTemporaryFile(mode = 'w+t')

    # initialize Doc2Vec model

    print('train Doc2Vec model')
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    train_corpus = [TaggedDocument(tokenizer(doc), [i]) for i, doc in enumerate(documents)]
    doc2vec_args["documents"] = train_corpus
    model = Doc2Vec(**doc2vec_args)

    # Word2Vec
    word_vectors = model.wv.get_normed_vectors()
    word_indexes = model.wv.key_to_index

    # create vocab
    vocab = list(model.wv.key_to_index.keys())

    # Doc2Vec
    document_vectors = model.dv.get_normed_vectors()

    # normalize vectors
    document_vectors = normalize(document_vectors)

    ngram_vocab_args = {'sentences': tokenized_corpus,
                        'min_count': 1,
                        'threshold': 1,
                        'delimiter': ' '}

    # phraser
    if use_phrases == True:
        print('detect phrases')
        from gensim.models.phrases import Phrases
        phrase_model = Phrases(**ngram_vocab_args)
        phrase_results = phrase_model.find_phrases(tokenized_corpus)
        phrases = list(phrase_results.keys())
        phrases_processed = [tokenizer(phrase) for phrase in phrases]

        phrase_vectors = np.vstack([model.infer_vector(doc_words=phrase,
                                                       alpha=0.025,
                                                       min_alpha=0.01,
                                                       epochs=100) for phrase in phrases_processed])

        # normalize
        phrase_vectors = normalize(phrase_vectors)

        # create word vectors
        word_vectors = np.vstack([word_vectors, phrase_vectors])

        # extent vocab
        vocab = vocab + phrases
    else:
        vocab = vocab
    # create word_indexes
    word_indexes = dict(zip(vocab, range(len(vocab))))

    vectorizer = CountVectorizer(tokenizer = return_doc, preprocessor = return_doc)
    doc_word_counts = vectorizer.fit_transform(tokenized_corpus)
    words = vectorizer.get_feature_names()
    word_counts = np.array(np.sum(doc_word_counts))
    vocab_inds = np.where(word_counts > 50)[0]

    return document_vectors, word_vectors, vocab, model

def documentReduction(company, data, document_vectors):

    """
    Alternative way of horizontal document reduction

    :param company:
    :param document_vectors:
    :return: company specific document vectors and corresponding indexes
    """

    data.reset_index(inplace = True, drop = True)

    from PreProcessing import getCompanyRanges
    company_ranges = getCompanyRanges(data, company)
    company_ranges

    company_vectors = document_vectors[company_ranges]

    return company_vectors, company_ranges

def calcTopicVectors(company_vectors, umap_neighbors, umap_embeddings,min_cluster_size, return_cluster = False):

    """
    Used to extract topic vectors out of company specific data

    :param company_vectors: company specific document vectors
    :param umap_neighbors: number of neighbours to approximate manifold M
    :param umap_embeddings: number of dimensions which are processed into HDBSCAN
    :param min_cluster_size: minimum amout of documents per cluster
    :param return_cluster: if set to true the cluster labels are shown
    :return: topic vectors, cluster labels (optional)
    """


    from umap import UMAP
    import hdbscan
    from Top2VecModule import _create_topic_vectors


    # reduce word embedding space via UMAP
    umap_args = {'n_neighbors': umap_neighbors,
                     'n_components': umap_embeddings,
                     'metric': 'cosine'}

    umap_model = UMAP(**umap_args).fit(company_vectors)

    # create hdbscan instance
    hdbscan_args = {'min_cluster_size': min_cluster_size,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'}

    cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)

    # calculate topic vectors
    topic_vectors = _create_topic_vectors(cluster.labels_, company_vectors)

    if return_cluster == False:
        return topic_vectors
    else:
        return topic_vectors, cluster.labels_


def productMining(product_list,data,normalize = True):
    """
    product mining algorithm for timely and sentiment based weighting of product proxies (px)

    :param product_list (str): list including names of respective products
    :param data (df): UGTD dataset
    :return: actuality and sentiment per px
    """

    from tqdm import tqdm
    import pandas as pd
    from SentimentAnalysis import flair_sentiment_scores
    import plotly.express as px
    import numpy as np
    from flair.models import TextClassifier

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # prepare input data
    data.reset_index(inplace = True, drop = True)
    data['text'] = data.text.astype(str)
    data['text'] = data.text.str.lower()

   # create flair instance for sentiment analysis
    sia = TextClassifier.load('en-sentiment')

    def flair_sentiment_scores(text, sia):

        from flair.data import Sentence
        sentence = Sentence(text)
        sia.predict(sentence)
        score = sentence.labels[0]
        if "POSITIVE" in str(score):
            return 1
        elif "NEGATIVE" in str(score):
            return -1
        else:
            return 0

    # initialize result frame
    results = pd.DataFrame(columns = ['product', 'score', 'importance', 'doc_count'], index = [i for i in range(len(product_list))])

    # iterate over each px and calc sentiment and actuality
    for position, product in enumerate(tqdm(product_list)):

        # get documents which include information about py
        product_documents = data[data.text.str.contains(product)]
        product_documents.reset_index(drop = True, inplace = True)

        # get product scores by applying sentiment analyis and aggregate by calculating the mean
        product_scores = product_documents.text.apply(lambda str: flair_sentiment_scores(str,sia))
        product_score = product_scores.mean()

        # get product importance by scaling dates and calculate actuality
        product_documents['date'] = pd.to_datetime(product_documents['date'])
        product_years = product_documents['date'].dt.year.unique()
        product_years = product_years[~np.isnan(product_years)]
        product_years.sort()

        year_scores = [i + 1 for i in range(len(product_years))]

        yearly_doc_count = product_documents.groupby('year').count()['text']
        importance_scores = yearly_doc_count * year_scores
        importance_score = importance_scores.mean()

        results.loc[position] = product, product_score, importance_score, product_documents.shape[0]


    if normalize == True:
       # results['score'] = NormalizeData(results['score'])
        results['importance'] = NormalizeData(results['importance']) + 1
       # results['doc_count'] = NormalizeData(results['doc_count'])


    results[['score', 'doc_count']] = results[['score', 'doc_count']].astype(float)

    # plot results
    fig = px.scatter(x = results['importance'], y = results['score'], text = results['product'], size = results['doc_count'], size_max = 100)
    fig.update_traces(textposition = 'top center')
    fig.update_layout(template = 'simple_white',  xaxis_title = 'Actuality $\Gamma$',  yaxis_title = 'Sentiment')
    fig.show()

    return results, fig

def normalizeData(data):
    import numpy as np

    return (data - np.min(data)) / (np.max(data) - np.min(data))


def getTopicTimeDistribution(topic_vectors, document_vectors, company_documents, word_vectors, vocab):

    """
    Time weighted topic modeling and sentiment evolvement analysis

    :param topic_vectors (array): topic vectors that wished to be analysed
    :param document_vectors (array): company specific documn vectors
    :param company_documents:
    :return: Plot showing topic and inherent mood development over time
    """

    # show topics over time
    import pandas as pd
    from Top2VecModule import _calculate_documents_topic
    from Top2VecModule import _find_topic_words_and_scores
    from Top2VecModule import flair_sentiment_scores
    import plotly.express as px
    from flair.models import TextClassifier

    # get topic words
    topic_words, topic_word_scores = _find_topic_words_and_scores(topic_vectors, word_vectors, vocab)
    doc_top, doc_dist = _calculate_documents_topic(topic_vectors, document_vectors)

    # assign topic to each document
    company_documents['Topic'] = ['Topic {}'.format(i) for i in doc_top]

    # get number of topics
    topics_df = pd.Series(doc_top).value_counts().index.sort_values()
    n_topics = len(topics_df)

    # aggregate topics based on year and topic number (t)
    year_topic_counts = pd.pivot_table(data=company_documents,
                                       values='text',
                                       index='year',
                                       columns='Topic',
                                       aggfunc='count',
                                       fill_value=0)

    year_topic_counts.columns = ['Topic {}'.format(i) for i in range(n_topics)]

    # exclude data for which no date is scraped or which only include little information in 2023
    try:
        try:
            year_topic_counts.drop([0,2023], axis = 0, inplace = True)
        except:
            year_topic_counts.drop(0, axis = 0, inplace = True)
    except:
        pass

    # attach years
    years = list(year_topic_counts.index.values) * n_topics
    year_topic_counts = year_topic_counts.melt()
    year_topic_counts['year'] = years

    # calc sentiment
    sia = TextClassifier.load('en-sentiment')

    # initialize sentiment and top words columns
    year_topic_counts['sentiment'] = 0
    year_topic_counts['top_words'] = 0

    # calculate sentiment for each year-topic combination
    for topic_year_combi in range(year_topic_counts.shape[0]):

        current_documents = company_documents[(company_documents['Topic'] == year_topic_counts['variable'].loc[topic_year_combi]) & (company_documents['year'] == year_topic_counts['year'].loc[topic_year_combi])]['text']
        sentiment = current_documents.apply(lambda str: flair_sentiment_scores(str,sia)).mean()
        year_topic_counts['sentiment'].loc[topic_year_combi] = sentiment

        # add topic words
        year_topic_counts['top_words'].loc[topic_year_combi] = topic_words[int(year_topic_counts['variable'].loc[topic_year_combi].split()[1])][:5]


    # add top words
    year_topic_counts['top_words'] = year_topic_counts['top_words'].astype(str)
    year_topic_counts['top_words'] = year_topic_counts['top_words'].str.replace('[','').str.replace(']', '')
    year_topic_counts['top_words']

    # plot results
    fig = px.scatter(year_topic_counts, 'year', 'top_words', size = 'value', color = 'sentiment', hover_data= ['top_words'])
    fig.update_layout(template = 'simple_white')
    fig.update_layout(xaxis=dict(domain=[0.15, 0.9]),
                      yaxis=dict(anchor='free', position=0.02,
                                 side='left'))


    fig.show()

    return year_topic_counts, fig


def radarAnalysis(model,company_name_1, company_name_2, data, keyword_sets, threshold, word_vectors, word_indexes,normalize = True):

    """
    Semantic competitor benchmarking

    :param model (instance): Doc2Vec model
    :param company_name_1 (str): name of company 1
    :param company_name_2 (str): name of company 2
    :param data (df): UGTD
    :param keyword_sets (list): list of words containing keywords
    :param threshold (int): lambda
    :param word_vectors (array): word embeddings
    :param word_indexes (array): word positions
    :param normalize (True | False): boolean
    :return: Radar graphs for respective companies and keywords aggregated via sentiment
    """

    import pandas as pd
    from SentimentAnalysis import vader_sentiment_scores
    from Top2VecModule import search_documents_by_keywords
    from tqdm import tqdm
    import numpy as np
    import plotly.graph_objects as go

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    results = []

    # iterate over each keyword
    for keyword_set in tqdm(keyword_sets):

        # get documents that contain
        documents, document_scores, document_ids = search_documents_by_keywords(model, [keyword_set], word_vectors, word_indexes,data.text)
        data.reset_index(inplace = True, drop = True)

        # get company scores (S_C)
        company_scores = pd.DataFrame(columns = ['company', 'document_id', 'score', 'text'], data = zip(data.loc[document_ids,'company'], document_ids, document_scores, data.loc[document_ids,'text']))

        # delete already used dfs and arrays for performance reasons
        del documents
        del document_scores
        del document_ids

        # only use documents whose S_C is above lambda
        company_scores = company_scores[company_scores['score'] > threshold]

        # seperate company 1 and compute sentiment
        company_1 = company_scores[company_scores['company'] == company_name_1]
        company_1 = company_1[company_1['score'] > threshold]
        company_1['sentiment'] = company_1['text'].apply(lambda str: vader_sentiment_scores(str))
        company_1 = company_1[company_1['sentiment'] != 0]

        # seperate company 2 and compute sentiment
        company_2 = company_scores[company_scores['company'] == company_name_2]
        company_2 = company_2[company_2['score'] > threshold]
        company_2['sentiment'] = company_2['text'].apply(lambda str: vader_sentiment_scores(str))
        company_2 = company_2[company_2['sentiment'] != 0]

        del company_scores

        # append results for given keyword
        results.append([company_1['sentiment'].mean(), company_2['sentiment'].mean()])

        del company_1
        del company_2

    if normalize == True:
        results = NormalizeData(results)

    # create radar graph
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[results[i][0] for i in range(len(results))],
        theta=keyword_sets,
        fill='toself',
        name=company_name_1
    ))
    fig.add_trace(go.Scatterpolar(
        r=[results[i][1] for i in range(len(results))],
        theta=keyword_sets,
        fill='toself',
        name=company_name_2
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                #range=[0, 5]
            )),
        showlegend=True
    )
    fig.show()

    return results, fig


def getProductProxy(brand_vector,word_vectors, word_indices):

    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    results = pd.DataFrame(columns = ['word', 'similarity'])

    for word_pos, word in tqdm(enumerate(word_indices.keys())):

        counter_vector = word_vectors[word_indices[word]].reshape(1,-1)
        similarity = cosine_similarity(brand_vector, counter_vector)
        results.loc[word_pos] = word, float(similarity)

    results.sort_values(by = 'similarity', inplace = True, ascending = False)

    return results

def flair_sentiment_scores(text, sia):

    from flair.models import TextClassifier
    from flair.data import Sentence
    sentence = Sentence(text)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return 1
    elif "NEGATIVE" in str(score):
        return -1
    else:
        return 0

def brandContextRecognition(company_vocab, comments, company,word_vectors,document_vectors,vocab, min_word_count,threshold,min_cluster_size,product_list = None):

    import pandas as pd
    from PreProcessing import textPreProcessing2
    # reduce company vocabulary

    if product_list:
        company_vocab = company_vocab.set_index('word').drop(product_list).reset_index(drop = False)

    # preprocess company comments
    company_docs = comments[comments['company'] == company]
    text = textPreProcessing2(company_docs,'text')

    # get word count from company vocabulary
    word_counts = pd.DataFrame(columns = ['word', 'count'])

    for word in company_vocab.word:
        sub_frame = pd.DataFrame(columns = ['word', 'count'])
        sub_frame.loc[0] = word, company_docs.text.str.contains(word).sum()
        word_counts = word_counts.append(sub_frame)

    word_counts['similarity'] = company_vocab['similarity'].values
    word_counts.sort_values(by = 'similarity', ascending = False).head(50)

    # filter again for similarity and count
    word_count_filt = word_counts[(word_counts['count'] > min_word_count) & (word_counts['similarity'] > threshold)]

    # filter company documents for company vocabulary
    company_data_filt = company_docs[company_docs.text.str.contains('|'.join(word_count_filt.word))]

    # perform topic_modeling
    brand_doc_vecs = document_vectors[company_data_filt.index]

    from Top2VecModule import calcTopicVectors
    topic_vectors = calcTopicVectors(brand_doc_vecs, 15,5,min_cluster_size)
    topics = len(topic_vectors)

    # get topic words
    from Top2VecModule import _find_topic_words_and_scores
    topic_words, topic_word_scores =  _find_topic_words_and_scores(topic_vectors, word_vectors, vocab)


    # assign documents to topic
    from Top2VecModule import _calculate_documents_topic
    doc_top, doc_dist = _calculate_documents_topic(topic_vectors, brand_doc_vecs, dist=True, num_topics=None)
    company_data_filt['topic'] = doc_top


    # assign topic to company_word
    word_count_filt['topic'] = 0
    word_count_filt['num_topic_occurence'] = 0
    word_count_filt.reset_index(drop = True, inplace = True)


    for word_pos,word in enumerate(word_count_filt.word):
        try:
            grouped_data = company_data_filt[company_data_filt['text'].str.contains(word)].groupby('topic').count()['text']
            max_top_count = grouped_data.max()
            top_index = grouped_data[grouped_data == max_top_count]
            word_count_filt.at[word_pos,'topic'] = top_index.index[0]
            word_count_filt.at[word_pos,'num_topic_occurence'] = company_data_filt[company_data_filt['topic'] == word_count_filt[word_count_filt['word'] == word].topic.values[0]].text.str.lower().str.contains(word).sum()

        except:
            pass


    word_count_filt['topic_words'] = word_count_filt['topic'].apply(lambda topic: str(topic_words[topic][:3].tolist()))
    word_count_filt['topic_words'] = word_count_filt['topic_words'].str.replace('[', '')
    word_count_filt['topic_words'] = word_count_filt['topic_words'].str.replace(']', '')
    word_count_filt['topic_words'] = word_count_filt['topic_words'].str.replace("'", '')
    import plotly.express as px

    fig = px.scatter(word_count_filt,'count', 'similarity', text = 'word', hover_data = ['word'], color = 'topic_words', size = 'num_topic_occurence')
    fig.show()

    return word_count_filt, fig

def hierarchical_topic_reduction(num_topics, topic_vectors, topic_sizes,document_vectors, word_vectors, vocab):
    """
    Reduce the number of topics discovered by Top2Vec.
    The most representative topics of the corpus will be found, by
    iteratively merging each smallest topic to the most similar topic until
    num_topics is reached.
    Parameters
    ----------
    num_topics: int
        The number of topics to reduce to.
    Returns
    -------
    hierarchy: list of ints
        Each index of hierarchy corresponds to the reduced topics, for each
        reduced topic the indexes of the original topics that were merged
        to create it are listed.
        Example:
        [[3]  <Reduced Topic 0> contains original Topic 3
        [2,4] <Reduced Topic 1> contains original Topics 2 and 4
        [0,1] <Reduced Topic 3> contains original Topics 0 and 1
        ...]
    """
    #self._validate_hierarchical_reduction_num_topics(num_topics)
    import numpy as np
    import pandas as pd
    from Top2VecModule import _calculate_documents_topic


    def _l2_normalize(vectors):
        from sklearn.preprocessing import normalize
        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]

    num_topics_current = topic_vectors.shape[0]
    top_vecs = topic_vectors
    top_sizes = [topic_sizes[i] for i in range(0, len(topic_sizes))]
    hierarchy = [[i] for i in range(topic_vectors.shape[0])]

    count = 0
    interval = max(int(document_vectors.shape[0] / 50000), 1)

    while num_topics_current > num_topics:

        # find smallest and most similar topics
        smallest = np.argmin(top_sizes)
        res = np.inner(top_vecs[smallest], top_vecs)
        sims = np.flip(np.argsort(res))
        most_sim = sims[1]
        if most_sim == smallest:
            most_sim = sims[0]

        # calculate combined topic vector
        top_vec_smallest = top_vecs[smallest]
        smallest_size = top_sizes[smallest]

        top_vec_most_sim = top_vecs[most_sim]
        most_sim_size = top_sizes[most_sim]

        combined_vec = _l2_normalize(((top_vec_smallest * smallest_size) +
                                      (top_vec_most_sim * most_sim_size)) / (smallest_size + most_sim_size))

        # update topic vectors
        ix_keep = list(range(len(top_vecs)))
        ix_keep.remove(smallest)
        ix_keep.remove(most_sim)
        top_vecs = top_vecs[ix_keep]
        top_vecs = np.vstack([top_vecs, combined_vec])
        num_topics_current = top_vecs.shape[0]

        # update topics sizes
        if count % interval == 0:
            doc_top = _calculate_documents_topic(topic_vectors=top_vecs,
                                                 document_vectors=document_vectors,
                                                 dist=False)
            topic_sizes = pd.Series(doc_top).value_counts()
            top_sizes = [topic_sizes[i] for i in range(0, len(topic_sizes))]

        else:
            smallest_size = top_sizes.pop(smallest)
            if most_sim < smallest:
                most_sim_size = top_sizes.pop(most_sim)
            else:
                most_sim_size = top_sizes.pop(most_sim - 1)
            combined_size = smallest_size + most_sim_size
            top_sizes.append(combined_size)

        count += 1

        # update topic hierarchy
        smallest_inds = hierarchy.pop(smallest)
        if most_sim < smallest:
            most_sim_inds = hierarchy.pop(most_sim)
        else:
            most_sim_inds = hierarchy.pop(most_sim - 1)

        combined_inds = smallest_inds + most_sim_inds
        hierarchy.append(combined_inds)

    # re-calculate topic vectors from clusters
    doc_top = _calculate_documents_topic(topic_vectors=top_vecs,
                                         document_vectors=document_vectors,
                                         dist=False)

    topic_vectors_reduced = _l2_normalize(np.vstack([document_vectors
                                                     [np.where(doc_top == label)[0]]
                                                    .mean(axis=0) for label in set(doc_top)]))

    #  hierarchy = hierarchy

    # assign documents to topic
    doc_top_reduced, doc_dist_reduced = _calculate_documents_topic(topic_vectors_reduced,
                                                                   document_vectors)
    # find topic words and scores
    topic_words_reduced, topic_word_scores_reduced = _find_topic_words_and_scores(topic_vectors_reduced, word_vectors, vocab)

    # calculate topic sizes
    #topic_sizes_reduced = _calculate_topic_sizes(hierarchy=True)

    return topic_vectors_reduced, doc_top_reduced, topic_words_reduced, topic_word_scores_reduced

def search_documents_by_keywords(model, keywords, word_vectors, word_indexes, documents):

    """

    Returns semantic filtering of documents given a keyword or set of keywords.

    :param model (Doc2Vec): model instance
    :param keywords (list): list of given keywords
    :param word_vectors (array): word embeddings
    :param word_indexes (array): word positions in V
    :param documents (df): UGTD
    :return: documents with descending order of the cosine similarity of the document and the keywords
    """

    import numpy as np
    document_ids = np.array(range(0, len(documents)))

    if len(keywords) != 0:
        word_vecs = word_vectors[[word_indexes[word] for word in keywords]]
    else:
        word_vecs = word_vectors[word_indexes[keywords]]

    sim_docs = model.docvecs.most_similar(positive=word_vecs, topn=len(documents))

    doc_indexes = [doc[0] for doc in sim_docs]
    doc_scores = np.array([doc[1] for doc in sim_docs])

    doc_ids = document_ids[doc_indexes]

    documents = documents[doc_indexes]

    return documents, doc_scores, doc_ids


def default_tokenizer(doc):
    # This part was copied from Top2Vec tokenizer, if you are using a specific tokenizer you should not use the default one when computing the measure
    """Tokenize documents for training and remove too long/short words"""
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import strip_tags

    return simple_preprocess(strip_tags(doc), deacc=True)


def search_vectors_by_vector(vectors, vector, num_res):
    ranks = np.inner(vectors, vector)
    indexes = np.flip(np.argsort(ranks)[-num_res:])
    scores = np.array([ranks[res] for res in indexes])

    return indexes, scores

def search_topics_by_vector(topic_vectors,vector, num_topics,topic_words, topic_scores):

    topic_nums, topic_scores_red = search_vectors_by_vector(topic_vectors,vector, num_topics)

    topic_words_reduced = [topic_words.copy()[topic] for topic in topic_nums]
    word_scores = [topic_scores.copy()[topic] for topic in topic_nums]

    return topic_words_reduced, word_scores, topic_scores_red, topic_nums

def query_topics(query, num_topics):

    tokenized_query = default_tokenizer(query)

    query_vec = model.infer_vector(doc_words=tokenized_query,
                                   alpha=0.025,
                                   min_alpha=0.01,
                                   epochs=100)

    return search_topics_by_vector(topic_vectors,query_vec, num_topics, topic_words, topic_word_scores)

def PWI_unigram(docs,doc_top,topic_words, num_words=20):
    import pandas as pd
    import numpy as np
    from nltk import FreqDist
    import math

    # This is used to tokenize the data and strip tags (as done in top2vec)
    tokenized_data = [default_tokenizer(doc) for doc in docs]
    # Computing all the word frequencies
    # First I concatenate all the documents and use FreqDist to compute the frequency of each word
    word_frequencies = FreqDist(np.concatenate(tokenized_data))

    # Computing the frequency of words per document
    # Remember to change the tokenizer if you are using a different one to train the model
    dict_docs_freqs = {}
    for i, doc in enumerate(docs):
        counter_dict = FreqDist(default_tokenizer(doc))
        if i not in dict_docs_freqs:
            dict_docs_freqs[i] = counter_dict

    PWI = 0.0
    p_d = 1 / len(docs)

    docs_doc_top = pd.DataFrame(columns =['doc', 'topic'], data = zip(docs,doc_top))
    pwi_per_topic = pd.DataFrame(columns = ['topic', 'PWI'], data = zip([i for i in range(len(topic_words))], [0 for i in range(len(topic_words))]))

    # This will iterate through the whole dataset and query the topics of each document.
    for i,doc in enumerate(docs):
        #    topic_words_query, word_scores_query, topic_scores_query, topic_nums = query_topics(query=doc, num_topics=num_topics)

        topic = docs_doc_top[docs_doc_top['doc'] == doc].topic.unique()[0]
        topic_words_query = topic_words[topic][:num_words]

        # Words of the topic
        # Topic scores is the topic importance for that document
        for word in topic_words_query:
            if word not in dict_docs_freqs[i]:
                # This is added just for some specific cases when we are using different collection to test
                continue
            # P(d,w) = P(d|w) * p(w)
            p_d_given_w = dict_docs_freqs[i].freq(word)
            p_w = word_frequencies.freq(word)
            p_d_and_w = p_d_given_w * p_w
            left_part = p_d_given_w #* t_score
            PWI += left_part * math.log(p_d_and_w / (p_w * p_d))
            pwi_per_topic.at[topic,'PWI'] = pwi_per_topic.at[topic,'PWI'] + left_part * math.log(p_d_and_w / (p_w * p_d))

    return PWI, pwi_per_topic

####################################
###   Additionaly functions    #####
### Used for further development ###
###     of Customer2Vec          ###
####################################

def getIndices(data, company_list, year):
    """

    :param data:
    :param company:
    :param year:
    :return:
    """
    import pandas as pd

    document_vector_indices = pd.DataFrame(columns = ['company', 'year','index'], data = zip(data['company'], data['year'], data.index))


    if len(company_list) == 1:
        indices = document_vector_indices[(document_vector_indices['company'] == company_list) & (document_vector_indices['year'] == year)].index.values
    else:

        indices = pd.DataFrame(columns = ['indices'])

        for company in company_list:

            indices = indices.append(pd.DataFrame(columns = ['indices'], data = document_vector_indices[(document_vector_indices['company'] == company) & (document_vector_indices['year'] == year)].index.values.ravel()))

    return indices


def getSharedVocab(data,vocab, word_vectors):

    """
    Function used to compute a shared vocabulary but not used now within Customer2Vec

    :param data (df): UGTD
    :param vocab (list): V
    :param word_vectors (array): word embeddings
    :return: shared vocabulary of given companies within UGTD
    """

    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics.pairwise import cosine_similarity

    shared_vocab = pd.DataFrame(columns = ['word', 'cosine_exceed', 'doc_count', 'company_count','use'], index = [word for word in range(len(vocab))])
    shared_vocab['cosine_exceed'] = 0
    shared_vocab['doc_count'] = 0
    shared_vocab['company_count'] = 0

    data['text'] = data['text'].str.lower()
    threshold = 0.5

    for pos,word in tqdm(enumerate(vocab)):
        shared_vocab['word'].loc[pos] = word
        shared_vocab['doc_count'].loc[pos] = data[data['text'].str.contains(f'(?:\s|^){word}(?:\s|$)')].shape[0]
        shared_vocab['company_count'].loc[pos] = len(data[data['text'].str.contains(f'(?:\s|^){word}(?:\s|$)')]['company'].unique())

        for word_check in vocab:
            vec_word = word_vectors[vocab.index(word)].reshape(1,-1)
            vec_check = word_vectors[vocab.index(word_check)].reshape(1,-1)
            cos_sim = cosine_similarity(vec_word,vec_check)

            if cos_sim > threshold:
                shared_vocab['cosine_exceed'].loc[pos] = shared_vocab['cosine_exceed'].loc[pos] + 1

            else:
                pass

    return shared_vocab


def aggregateData(company, comments, posts):

    import pandas as pd

    # prepare
    company_comments = comments[comments['company'] == company]
    company_comments['date'] = pd.to_datetime(company_comments['date'])
    company_comments = company_comments.rename(columns = {'text':'comment_text'})

    company_comments = company_comments[['company','comment_text','date','source','post_id']]
    company_comments = company_comments.drop_duplicates()

    company_posts = posts[posts['company'] == company][['company','post_id','text','source','likes']]
    company_posts = company_posts.drop_duplicates()
    sources = company_posts.source.unique()
    
    company_comments = company_comments[company_comments['source'].isin(sources)]

    company_comments.set_index(['company','post_id','source'],inplace = True)
    company_posts.set_index(['company','post_id','source'],inplace = True)

    company_comments = company_comments.join(company_posts)
    company_posts.reset_index(drop = False, inplace = True)

    max_posts = pd.DataFrame(columns = ['date', 'text', 'comment_count', 'likes_post'])
    counter = 0

    indexes = list(company_comments.date.unique().astype(str))

    for date in indexes:
        try:
            comments_date = company_comments[company_comments['date'] == date[:10]]
            comment_count = comments_date.groupby('post_id').count()['text']
            max_count = comment_count[comment_count == comment_count.max()]
            max_post_text = company_posts[company_posts['post_id'] == max_count.index[0]].text.values[:50]
            max_post_likes = company_posts[company_posts['post_id'] == max_count.index[0]].likes.sum()
            max_posts.loc[counter] = date[:10], max_post_text, int(max_count.index[0]), max_post_likes
            counter = counter + 1
        except:
            pass

    max_posts['date'] = pd.to_datetime(max_posts['date'])
    max_posts.set_index('date', inplace = True)
    max_posts.index = max_posts.index.to_period('M').to_timestamp('M')
    max_posts.reset_index(inplace = True, drop = False)
    max_posts.drop_duplicates('date', keep = 'last', inplace = True)
    max_posts.set_index('date', inplace = True)
    max_posts = max_posts.rename(columns = {'text':'post_text'})

    company_comments.reset_index(inplace = True, drop = False)
    company_comments.set_index('date',inplace = True)
    company_comments = company_comments.resample('M').count()

    final_data = company_comments.join(max_posts[['post_text','likes_post']])

    final_data.reset_index(drop = False, inplace = True)

    max_data = final_data[final_data['text'] > final_data['text'].quantile(0.75)]
    max_data['post_text'] = max_data['post_text'].apply(lambda str: str[:20])

    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.graph_objs import Layout

    fig_1 = px.line(final_data, 'date','comment_text', color_discrete_sequence=['black'], title = f'{company} social media history')
    fig_2 = px.scatter(max_data,'date','comment_text', size = [10 for i in range(max_data.shape[0])], color_discrete_sequence=['grey'],
                       hover_data = ['post_text'], color = 'likes_post')

    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


    fig_3 = go.Figure(data = fig_1.data + fig_2.data, layout = layout)

    return final_data, max_data, fig_3

def getTextDistribution(data,company_list, normalize = True):

    import pandas as pd
    from Top2VecModule import aggregateData
    from Top2VecModule import normalizeData


    comp_total = pd.DataFrame(columns = ['year', 'text', 'company'])


    for company in company_list:

        comp = aggregateData(data,company).drop('company').reset_index(drop = False)
        comp['company'] = company

        if normalize == True:
            comp['text'] = normalizeData(comp['text'].astype(int))

        comp_total = comp_total.append(comp)

    return comp_total

def namedEntityRecognition(data):

    from tqdm import tqdm
    import en_core_web_sm
    nlp = en_core_web_sm.load()

    data['text'] = data['text'].astype(str)

    for text in tqdm(range(data.shape[0])):

        doc = nlp(data['text'][text])
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        for entity in entities:

            if entity[1] == 'PERSON' and entity[0] != data['company'][text]:

                data['text'][text] = data['text'][text].replace(entity[0],'')
            else:
                pass

    return data

def eventMining(posts, comments, events,event_indices,threshold,post_vectors):

    from flair.models import TextClassifier
    sia = TextClassifier.load('en-sentiment')

    import en_core_web_sm
    nlp = en_core_web_sm.load()

    import pandas as pd

    posts['comment_count'] = 0
    posts['sentiment'] = 0
    posts['event_category'] = 0
    #events = ['PRODUCT','ORG', 'CARDINAL', 'FAC']

    def flair_sentiment_scores(text, sia):

        from flair.models import TextClassifier
        from flair.data import Sentence
        sentence = Sentence(text)
        sia.predict(sentence)
        score = sentence.labels[0]
        if "POSITIVE" in str(score):
            return 1
        elif "NEGATIVE" in str(score):
            return -1
        else:
            return 0

    document_vectors = post_vectors

    posts.reset_index(drop = True, inplace = True)
    #event_indices = [10,11,13,22,80,138,329,518]

    event_vector = document_vectors[event_indices].mean(axis = 0)

    from sklearn.metrics.pairwise import cosine_similarity
    vec_1 = event_vector.reshape(1,-1)

    event = []
    no_event = []

    # event and no event
    for vec in range(len(document_vectors)):

        vec_2 = document_vectors[vec].reshape(1,-1)
        similarity = cosine_similarity(vec_1, vec_2)

        if similarity > threshold:
            event.append(vec)
        else:
            no_event.append(vec)

    print(f'found {len(event)} events matching to event vector')

    # event general
    for vec in range(len(document_vectors)):

        vec_2 = document_vectors[vec].reshape(1,-1)
        similarity = cosine_similarity(vec_1, vec_2)

        posts.at[vec,'event_similarity'] = similarity[0][0]

    # attach sentiment and comment count
    for post in range(posts.shape[0]):
        posts['comment_count'].loc[post] = comments[comments['post_id'] == posts['post_id'].loc[post]].shape[0]

        if posts['event_similarity'].loc[post] > threshold:
            sentiment_corpus = comments[comments['post_id'] == posts['post_id'].loc[post]].text
            sentiment_corpus['sentiment'] = sentiment_corpus.apply(lambda str: flair_sentiment_scores(str,sia))
            posts['sentiment'].loc[post] = sentiment_corpus['sentiment'].mean()

            doc = nlp(posts['text'].loc[post])
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            ents = []

            for ent in entities:

                if ent[1] in events:
                    ents.append(ent[0])
                else:
                    pass

            if ents == []:
                continue
            else:
                posts['event_category'].loc[post] = ents


    events = posts[posts['event_similarity'] >= threshold]

    events['time'] = pd.to_datetime(events['time'])
    events['month'] = events['time'].dt.month
    events['year'] = events['time'].dt.year
    event_text = events[['event_category', 'month', 'year']]
    event_text['event_category'] = event_text.groupby(['year', 'month'])['event_category'].transform(lambda x: ','.join(x.astype(str)))
    event_text.set_index(['year','month'], inplace = True)

    events_agg = events[['likes', 'event_similarity', 'sentiment', 'comment_count', 'time','month','year']]
    events_agg['time'] = pd.to_datetime(events_agg['time'])
    events_agg.set_index('time', inplace = True)

    agg = events_agg.resample('M').mean()
    agg['sentiment'].fillna(0,inplace = True)
    agg.reset_index(drop = False, inplace = True)
    agg.set_index(['year','month'],inplace = True)

    agg.dropna(inplace = True)

    agg = agg.join(event_text)

    import plotly.express as px
    px.scatter(agg, 'time','event_similarity', hover_data = ['event_category'], color = 'sentiment', size = 'comment_count', size_max = 100,color_continuous_scale='greens')

    return agg
