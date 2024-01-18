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

# perform model

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 50, min_count = 50, vector_size = 400, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e50_m50_v400')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 50, min_count = 50, vector_size = 300, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e50_m50_v300')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 50, min_count = 50, vector_size = 200, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e50_m50_v200')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 400, min_count = 50, vector_size = 300, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e400_m50_v300')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 400, min_count = 50, vector_size = 200, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e400_m50_v200')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 400, min_count = 50, vector_size = 400, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e400_m50_v400')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 200, min_count = 50, vector_size = 400, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e200_m50_v400')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 200, min_count = 50, vector_size = 200, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e200_m50_v200')

from Top2VecModule import top2vecReBuild
document_vectors, word_vectors, vocab, model = top2vecReBuild(documents = comments_red.text, epochs = 200, min_count = 50, vector_size = 300, use_phrases = False)
model.save('vector_space_total_deeplearn_raw_e200_m50_v300')


# check for consitency

# choose companies
companies = comments_red.company.unique()

# choose parameters
epochs = [50,400,200]
vector_sizes = [200,300,400]

check = pd.DataFrame(columns = ['e50_m50_v400', 'e50_m50_v300', 'e50_m50_v200','e400_m50_v300',
                                'e400_m50_v200', 'e400_m50_v400', 'e200_m50_v400', 'e200_m50_v200', 'e200_m50_v300', 'shape'], index = [company for company in comments_red.company.unique()])

for company in companies:
    for epoch in epochs:
        for size in vector_sizes:

            try:
                results = pd.read_excel(f'final_results/{company}_PIW_simulation_raw_e{epoch}_m50_v{size}.xlsx')
                check.at[company, f'e{epoch}_m50_v{size}'] = 1
                check.at[company, 'shape'] = results.shape

            except:
                check.at[company, f'e{epoch}_m50_v{size}'] = 0
                check.at[company, 'shape'] = 0








