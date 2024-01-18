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
model_syntax = 'raw_e50_m50_v300'
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

remove = 1

remove_words = words[words['cluster'] == remove].word.values
remove_vectors = list(words[words['cluster'] == remove].index)

import numpy as np
word_vectors = np.delete(word_vectors,remove_vectors, axis = 0)
vocab = np.delete(np.array(vocab),remove_vectors, axis = 0)
vocab = vocab.tolist()

# pat = r'\b(?:{})\b'.format('|'.join(remove_words))

# comments_red['text_removed'] = comments_red['text'].str.replace(pat, '')

# from Top2VecModule import top2vecReBuild
# document_vectors, word_vectors, vocab, model = top2vecReBuild(comments_red.text,40, use_phrases=True)
# model.save('semifinal_deeplearn_meijer15_nowehkamp_wordcount0')

# loop over topics and find

# choose company and get attributes
from tqdm import tqdm
companies = pd.DataFrame(columns = ['company', 'topics'])
num_topics = 20 #(1) #1000 #(2)

comments_red['topic'] = 0

for comp_count,company in tqdm(enumerate(comments_red.company.unique())):

    print(f'Getting topics of {company}')

    company_documents = comments_red[comments_red['company'] == company]

    # create company_specific vectors d and w
    #from Top2VecModule import documentReduction

    company_vectors = document_vectors[company_documents.index]

    check_voc = pd.DataFrame(vocab)
    check_voc['isin_doc'] = 0

    for idx, word in enumerate(vocab):
        for doc in company_documents.text:

            if word in doc:
                check_voc.at[idx,'isin_doc'] = 1

    company_vocab = [vocab[i] for i in check_voc[check_voc['isin_doc'] == 1].index.values]

    remove_vectors = list(check_voc[check_voc['isin_doc'] == 0].index.values)
    company_word_vectors = np.delete(word_vectors,remove_vectors, axis = 0)

    from Top2VecModule import calcTopicVectors
    from Top2VecModule import _find_topic_words_and_scores

    # mymuesli 15,15,150 -> 15 topics
    # Birkenstock 15,5,220
    # meijer 15,5220
    topic_vectors = calcTopicVectors(company_vectors, 15,5,15) # ToDo: PMI based min_cluster_size optimization
    topic_words, topic_scores = _find_topic_words_and_scores(topic_vectors, company_word_vectors,company_vocab)

    # reduce topic amount
    from Top2VecModule import hierarchical_topic_reduction
    from Top2VecModule import _calculate_documents_topic

    doc_top, doc_dist = _calculate_documents_topic(topic_vectors, company_vectors, dist=True, num_topics=None)
#   comments_red[comments_red['company'] == company].at[:,'topic'] = doc_top

    topic_sizes = pd.Series(doc_top).value_counts()
    topic_vectors, doc_top, topic_words, topic_word_scores = hierarchical_topic_reduction(num_topics, topic_vectors, topic_sizes,company_vectors, company_word_vectors, company_vocab)

    if comp_count == 0:
            topic_vectors_all = topic_vectors
            topic_words_all = topic_words
        else:
            topic_vectors_all = np.vstack((topic_vectors_all,topic_vectors))
            topic_words_all = np.vstack((topic_words_all,topic_words))

    companies.loc[comp_count] = company, len(topic_words)
    print(f'{company} completed')

companies_idx = []
topics_idx = []

for company in companies.company:

    companies_idx.extend(companies[companies['company'] == company].company.values[0] for i in range(companies[companies['company'] == company].topics.values[0]))
    topics_idx.extend(i for i in range(companies[companies['company'] == company].topics.values[0]))


# assign index
companies_idx = pd.DataFrame(columns = ['company', 'topic'], data = zip(companies_idx, topics_idx))
local_topic_relation = pd.DataFrame()


from sklearn.metrics.pairwise import cosine_similarity

for idx_1,topic_vector_1 in enumerate(topic_vectors_all):

    company_1 = companies_idx.loc[idx_1].company
    topic_1 = companies_idx.loc[idx_1].topic

    for idx_2, topic_vector_2 in enumerate(topic_vectors_all):

        company_2 = companies_idx.loc[idx_2].company
        topic_2 = companies_idx.loc[idx_2].topic

        cs = cosine_similarity(topic_vector_1.reshape(1,-1), topic_vector_2.reshape(1,-1))
        local_topic_relation.loc[f'{company_1}_{topic_1}',f'{company_2}_{topic_2}'] = float(cs)


def topic_finder(company,topic,companies_idx):

    pos = companies_idx[(companies_idx['company'] == company) & (companies_idx['topic'] == topic)].index
    return pos[0]

results = pd.DataFrame(columns = ['company_1', 'company_2', 'topic_1', 'topic_2', 'topic_words_1', 'topic_words_2', 'similarity'])
counter = 0

for company in companies.company:
    for topic in range(companies[companies['company'] == company].topics.values[0]):
        #print(company)
        try:
                res = local_topic_relation.loc[f'{company}_{topic}'].sort_values(ascending= False)
                cos = res[1]
                company_1 = company
                company_2 = res.index[1].split('_')[0]

                if company_1 == company_2:
                    continue
                else:

                    topic_1 = topic
                    topic_2 = res.index[1].split('_')#[1]

                    if len(topic_2) == 3:
                        topic_2 = res.index[1].split('_')[2]
                    else:
                        topic_2 = res.index[1].split('_')[1]

                    topic_words_1 = topic_words_all[topic_finder(company_1,int(topic_1),companies_idx)][:20]
                    topic_words_2 = topic_words_all[topic_finder(company_2,int(topic_2),companies_idx)][:20]


                    results.loc[counter] = company_1, company_2, topic_1, topic_2, topic_words_1, topic_words_2, cos
                    counter = counter + 1
        except:
            pass

