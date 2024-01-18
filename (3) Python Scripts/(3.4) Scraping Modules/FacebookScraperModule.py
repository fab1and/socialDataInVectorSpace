#!pip install facebook-scraper
#!pip install facebook_scraper --upgrade
import facebook_scraper
from facebook_scraper import get_posts

import pandas as pd
from tqdm import tqdm
# set working directory in which
import os
os.chdir("C:/Users/Fabia/pythoninput")

i = 0


options_set = {
    "comments": True,
    #"posts_per_page": 10000,
    "allow_extra_requests": True,
    # "reactors": True,
    "progress": True
}

comments_uncleaned = pd.DataFrame(columns = ['company','comment_id', 'post_id','text', 'time'])
posts_uncleaned = pd.DataFrame(columns = ['company', 'post_id', 'text','likes', 'time'])

company_list = ['tado° - Smart Thermostats', 'Getaround', 'N26', '']

company = 'SennheiserDeutschland' # #NeemansOfficial

post_id = 0
comments_id = 0

#cookies = ['cookies_1.txt', 'cookies_2.txt']
file = 'cookies.json'#cookies[random.randint(0, 1)]
#set_cookies(file)

#try:
# download posts
posts = list(get_posts(company,options=options_set, cookies = file, pages = None))

# download comments per post

for post in tqdm(range(len(posts))):

    current_post = posts[post]
    try:
        posts_uncleaned.loc[post] = company, post_id, current_post['text'].replace('\\n', '').replace('\n','').replace(',', ''),current_post['likes'], current_post['time']
    except:
        posts_uncleaned.loc[post] = company, post_id, current_post['text'], current_post['likes'], current_post['time']

    comments_id = 0
    post_id = post_id + 1

    for current_comment in range(len(current_post["comments_full"])):
        #try:
        comments_uncleaned.loc[i] = company, comments_id, post_id, current_post["comments_full"][current_comment]['comment_text'].replace('\\n', '').replace('\n','').replace(',', ''), current_post['time']
        #except:
        #    comments_uncleaned.loc[i] = company, comments_id, post_id, current_post["comments_full"][current_comment]['comment_text'], current_post['time']
        print(current_post["comments_full"][current_comment]['comment_text'][:50])
        print(i)
        i += 1
        comments_id = comments_id + 1

comments_uncleaned['source'] = 'facebook'
posts_uncleaned['source'] = 'facebook'
comments_uncleaned.to_csv(f'C:/Users/Fabia/pythoninput/comments_{company}_facebook_re.csv')
posts_uncleaned.to_csv(f'C:/Users/Fabia/pythoninput/posts_{company}_facebook_re.csv')
#%%
