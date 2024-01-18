import pandas as pd
import os
os.chdir('C:/Users/Fabia/DataspellProjects/Data Generating Process')

# load complete data
comments = pd.read_excel('comments_final_2005.xlsx')

# get word count
comments['text'] = comments['text'].astype(str)
comments['word_count'] = comments.text.apply(lambda str: len(str.split()))
comments_red = comments[comments['word_count'] > 2]
comments_red.sort_values(by = 'company', inplace = True)
comments_red.reset_index(inplace = True, drop = True)

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from nltk import FreqDist
import math

from gensim.models.doc2vec import Doc2Vec
model_syntax = 'raw_e400_m50_v400'
model = Doc2Vec.load(f'vector_space_total_deeplearn_{model_syntax}')
word_vectors = model.wv.get_normed_vectors()
word_indexes = model.wv.key_to_index

# create vocab
vocab = list(model.wv.key_to_index.keys())

# Doc2Vec
document_vectors = model.dv.get_normed_vectors()

# PreProcessing of word vectors
from Top2VecModule import calcTopicVectors
word_topic_vecs, words = calcTopicVectors(word_vectors, 15, 5,40, return_cluster = True)
words = pd.DataFrame(columns = ['word', 'cluster'], data = zip(vocab,words))

remove = 0

remove_words = words[words['cluster'] == remove].word.values
remove_vectors = list(words[words['cluster'] == remove].index)

import numpy as np
word_vectors = np.delete(word_vectors,remove_vectors, axis = 0)
vocab = np.delete(np.array(vocab),remove_vectors, axis = 0)
vocab = vocab.tolist()


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

    return PWI


done_list = []
done_list.append('meijer')

for company in comments_red.company.unique():

    try:

        pd.read_excel(f'{company}_PIW_simulation_{model_syntax}.xlsx')
        done_list.append(company)
    except:
        pass

import tqdm as tqdm

for company in comments_red.company.unique():

    if company in done_list:
        continue

    print(f'Calculating PWI for {company}')

    num_topics = 20

    # choose company and get attributes
    company_documents = comments_red[comments_red['company'] == company]

    #from Top2VecModule import documentReduction
    #company_vectors, company_ranges = documentReduction(company, comments_red, document_vectors)

    company_vectors = document_vectors[company_documents.index]

    from Top2VecModule import calcTopicVectors
    from Top2VecModule import _find_topic_words_and_scores
    from Top2VecModule import hierarchical_topic_reduction
    from Top2VecModule import _calculate_documents_topic


    check_voc = pd.DataFrame(vocab)
    check_voc['isin_doc'] = 0

    for idx, word in enumerate(vocab):
        for doc in company_documents.text:

            if word in doc:
                check_voc.at[idx,'isin_doc'] = 1

    company_vocab = [vocab[i] for i in check_voc[check_voc['isin_doc'] == 1].index.values]

    remove_vectors = list(check_voc[check_voc['isin_doc'] == 0].index.values)
    company_word_vectors = np.delete(word_vectors,remove_vectors, axis = 0)

    # mymuesli 15,15,150 -> 15 topics
    # Birkenstock 15,5,220
    # meijer 15,5220

    n_neighbors = [5,7,10,15,20]
    n_components = [5,10,15]

    # define results frame and counter index
    simulation_results = pd.DataFrame(columns = ['n_neighbors', 'n_components', 'PWI', 'topic_count'])
    idx = 0

    # define corpus
    docs = company_documents.text

    from tqdm import tqdm
    for n_neighbor in tqdm(n_neighbors):
        for n_component in tqdm(n_components):
            try:
                # get t for min cluster size = 15
                topic_vectors = calcTopicVectors(company_vectors, n_neighbor,n_component,15)
                topic_words, topic_scores = _find_topic_words_and_scores(topic_vectors, company_word_vectors, company_vocab)
                doc_top, doc_dist = _calculate_documents_topic(topic_vectors, company_vectors, dist=True, num_topics=None)
                topic_sizes = pd.Series(doc_top).value_counts()

                # reduce topics to num_topics
                topic_vectors, doc_top, topic_words, topic_word_scores = hierarchical_topic_reduction(num_topics, topic_vectors, topic_sizes,company_vectors, company_word_vectors, company_vocab)

                # calculate topics based on num_topics
                PWI_uni = PWI_unigram(docs,doc_top,topic_words,num_words=20)
                simulation_results.loc[idx] = n_neighbor, n_component, PWI_uni, len(topic_vectors)
                #simulation_results.to_excel('simulation_results_fractal.xlsx')
                print(f'{PWI_uni}')
                idx = idx + 1
            except:
                pass

    simulation_results.to_excel(f'{company}_PIW_simulation_{model_syntax}.xlsx')
