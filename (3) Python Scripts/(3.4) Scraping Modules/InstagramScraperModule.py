#login

import os
import pandas as pd
from random import randint
os.getcwd()

data = pd.read_csv('C:/Users/Fabia/pythoninput/comments_wehkampnl_facebook.csv')

# stuff that has to be initialized external

account_set = pd.read_csv('account_set.csv', sep = ';')#.loc[randint(0,pd.read_csv('account_set.csv', sep = ';').shape[0]].transpose()

comment_id = 0

skip_strs = ['gewinnspiel', 'verlosen', 'giveaway', 'zu gewinnen', 'give away', 'gewinner', 'viel glück', 'giving away']

scraping_frame = pd.DataFrame(columns = ['company_index','company','URL_index','URL','done'] )

def appendScrapingUrls(scraping_frame, company):

    new_data = pd.read_csv(f'{company}_urls.csv').drop('Unnamed: 0', axis = 1)

    temp_frame = pd.DataFrame(columns = ['company_index','company','URL_index','URL','done'], index = [row for row in range(new_data.shape[0])])

    try:

        temp_frame['company_index'] = max(scraping_frame['company_index'].unique()) + 1
    except:
        pass

    temp_frame['company'] = str(company)
    temp_frame['URL_index'] = new_data.index
    temp_frame['URL'] = new_data.values
    temp_frame['done'] = 0

    scraping_frame = scraping_frame.append(temp_frame)

    return scraping_frame

company_list = ['ledlenser_official']

for company in company_list:

    scraping_frame = appendScrapingUrls(scraping_frame,company)

scraping_frame.to_csv('scraping_frame_cyb.csv')

def appendRow(dataframe, List):

    from csv import writer
    with open(f'{dataframe}.csv', 'a', newline = '',encoding='utf-8-sig') as f_object:

        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object, delimiter = ';')

        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(List)
        # Close the file object
        f_object.close()

def instagramLogin():
    """
    Cycles through username and password combinations to avoid getting banned.
    :return: browser window with logged in account
    """

    from itertools import cycle
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.common.by import By
    import os
    import time
    from selenium.webdriver.support import expected_conditions as EC
    from datetime import datetime
    from random import randint
    import pandas as pd

    user_account = pd.read_csv('account_set.csv', sep = ';').loc[randint(0,pd.read_csv('account_set.csv', sep = ';').shape[0]-1)].transpose()

    driver = webdriver.Chrome()#(options = options)
    #current_account_number = next(switching_accounts)

    import threading

    username_input = user_account['username']
    password_input = user_account['password']

    print(f'{username_input} is used for dgp')

    options = Options()
    #options.add_argument('--headless')

    driver.get("https://www.instagram.com/")

    # wait until cookies appear
    wait = WebDriverWait(driver, 10)

    # allow only necessary cookies by clicking the button

    #button = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[4]/div/div/button[1]')))
    # button = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[4]/div/div/button[1]')))
    #driver.execute_script("arguments[0].click()", button)

    time.sleep(randint(5,10))
    cookies = driver.find_element(By.XPATH, "//button[contains(text(), 'Nur erforderliche Cookies erlauben')]").click()


    # login
    time.sleep(randint(5,40))
    username = driver.find_element(By.CSS_SELECTOR, "input[name='username']")
    password = driver.find_element(By.CSS_SELECTOR, "input[name='password']")
    username.clear()
    password.clear()
    username.send_keys(username_input)
    password.send_keys(password_input)
    login = driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    # do not save login_information
    time.sleep(randint(10,30))

    try:
        notnow = driver.find_element(By.XPATH, "//button[contains(text(), 'Jetzt nicht')]").click()
    except:
        pass

    # do not allow notifications
    time.sleep(randint(10,30))
    try:
        notnow2 = driver.find_element(By.XPATH, "//button[contains(text(), 'Jetzt nicht')]").click()
    except: pass

    #return driver and pass to other functions
    return driver, user_account

def searchCompany(driver, company):
    """
    enters the company name into the search box and selects the matching entity
    :param company_name: str
    :return:
    """
    import time
    from selenium.webdriver.common.by import By
    from random import randint

    try:
        time.sleep(randint(5,10))

        searchbox = driver.find_element(By.CSS_SELECTOR, '[aria-label="Sucheingabe"]')
        searchbox.clear()
        searchbox.send_keys(company)  # insert company names here
        time.sleep(randint(10,20))

        name_classes = driver.find_elements(By.CLASS_NAME, '_abm4')

        correct_name = 0

        for name in range(len(name_classes)):
            try:
                comp = name_classes[name].text.split('\n')[1]
            except:
                comp = name_classes[name].text.split('\n')

            if comp == company:
                break
            else:
                correct_name = correct_name + 1

        correct_name = name_classes[correct_name]
        correct_name.click()

    except:
        loop = driver.find_element(By.CSS_SELECTOR, '[aria-label="Suche"]').click()
        searchbox = driver.find_element(By.CSS_SELECTOR, '[aria-label="Sucheingabe"]')
        searchbox.clear()
        searchbox.send_keys(company)  # insert company names here

        time.sleep(randint(5,12))
        name_classes = driver.find_elements(By.CLASS_NAME, '_abm4')
        correct_name = 0

        for name in range(len(name_classes)):
            try:
                comp = name_classes[name].text.split('\n')[1]
            except:
                comp = name_classes[name].text.split('\n')

            if comp == company:
                break
            else:
                correct_name = correct_name + 1

        correct_name = name_classes[correct_name]
        correct_name.click()

def fetchPosts(driver):

    import time
    from selenium.webdriver.common.by import By
    from random import randint

    time.sleep(randint(8,20))
    urls = []

    scrolldown=driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var scrolldown=document.body.scrollHeight;return scrolldown;")
    match=False
    while(match==False):
        print("Scraper is scrolling...")
        last_count = scrolldown
        time.sleep(randint(5,14))
        links = driver.find_elements(By.TAG_NAME, value = "a")
        for link in links:
            post = link.get_attribute('href')
            if '/p/' in post:
                urls.append(post)
        scrolldown = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var scrolldown=document.body.scrollHeight;return scrolldown;")

        if last_count==scrolldown:
            match=True

    return urls

def loadComments(driver):

    import time
    from selenium.webdriver.common.by import By
    import pandas as pd
    from random import randint

    counter = 0

    try:
        button_exists = True

        while button_exists == True:

            time.sleep(randint(5,20))
            load_more_comments = driver.find_element(By.CSS_SELECTOR, "[aria-label='Weitere Kommentare laden']").click()
            comments = driver.find_elements(By.CLASS_NAME, value='_a9ym')

            comment_texts = []

            for comment in comments[-10:]:

                comment_texts.append(comment.text.split('\n'))

            comment_texts = pd.DataFrame(comment_texts).drop_duplicates()

            if comment_texts.shape[0] < 10:
                counter = counter + 1

            else:
                continue

            if counter == 3:
                button_exists = False

            else:
                continue


            if len(comments) > 500:

                button_exists = False

        else:

            button_exists = False
    except:
        button_exists = False

def scrapePost(driver,post_id,company):

    from selenium.webdriver.common.by import By
    import pandas as pd
    from datetime import date

    posts_uncleaned = pd.DataFrame(columns=['company', 'post_id', 'text', 'likes', 'time', 'timestamp'])

    try:
        post_time = driver.find_elements(By.CLASS_NAME, '_aaqe')[0].text
        try:
            likes_post = int(driver.find_elements(By.CLASS_NAME, '_ae5m')[0].text.split(' ')[1].replace(',', ''))
        except:
            likes_post = int(driver.find_elements(By.CLASS_NAME, '_ae5m')[0].text.split(' ')[0].replace('.', ''))
        post_text = driver.find_elements(By.CLASS_NAME, '_a9zs')[0].text.replace('\n','')
        timestamp = str(date.today())
        posts_uncleaned.loc[post_id] = company, post_id, post_text, likes_post, post_time, timestamp
        post_id += 1

    except:
        post_time = driver.find_elements(By.CLASS_NAME, '_aaqe')[0].text
        post_text = driver.find_elements(By.CLASS_NAME, '_a9zs')[0].text.replace('\n','')
        try:
            likes_post = int(driver.find_elements(By.CLASS_NAME, '_ae5m')[0].text.split(' ')[1].replace(',', ''))
        except:
            likes_post = int(driver.find_elements(By.CLASS_NAME, '_ae5m')[0].text.split(' ')[1].replace('.', ''))
        timestamp = str(date.today())
        posts_uncleaned.loc[post_id] = company, post_id, post_text, likes_post, post_time, timestamp
        post_id += 1

    appendRow('posts',list(posts_uncleaned.iloc[0].to_list()))

    return post_text

def scrapeComments(driver,comment_id,post_id,company):

    from selenium.webdriver.common.by import By
    import pandas as pd
    from datetime import date
    comments = driver.find_elements(By.CLASS_NAME, value='_a9ym')

    timestamp = str(date.today())

    for comment in comments:
        comments_uncleaned = pd.DataFrame(columns=['company', 'post_id','verified', 'text', 'likes', 'time', 'timestamp'])

        try:

            if comment.text.split('\n')[1] == "Verifiziert":

                try:
                    text = comment.text.split('\n')
                    likes = str([like_count for like_count in text if ('Gefällt') in like_count]).split('Gefällt')[1]
                except:
                    likes = 0
                    comments_uncleaned.loc[0] = company, post_id,comment.text.split('\n')[1], comment.text.split('\n')[2], likes, comment.text.split('\n')[3], timestamp
                    comment_id += 1

                appendRow('comments',list(comments_uncleaned.loc[0].to_list()))

            else:

                text = comment.text.split('\n')
                try:
                    likes = str([like_count for like_count in text if ('Gefällt') in like_count]).split('Gefällt')[1]
                except:
                    likes = 0
                comments_uncleaned.loc[0] = company, post_id,  'not verified', comment.text.split('\n')[1], likes, comment.text.split('\n')[2], timestamp
                comment_id += 1

                appendRow('comments',list(comments_uncleaned.loc[0].to_list()))

        except:
            pass

def subScrapingFrame(scraping_frame, companies, samples):

    tempscraping_frame = scraping_frame.copy()
    subscraping_frame = pd.DataFrame(columns = ['company_index', 'company', 'URL_index', 'URL', 'done'])

    for compcount, company in zip(range(len(company_list)), company_list):

        subscraping_frame_2 = tempscraping_frame[tempscraping_frame['done'] == 0]

        try:
            subscraping_frame_2 = subscraping_frame_2[subscraping_frame_2['company'] == company]
        except:
            continue

        comp_count = subscraping_frame_2.shape[0]
        samples_adj = samples

        while comp_count < samples_adj:

            samples_adj = samples_adj - 1

        subscraping_frame_2 = subscraping_frame_2.sample(samples_adj)
        subscraping_frame = subscraping_frame.append(subscraping_frame_2)

    return subscraping_frame

def connect(host='http://google.com'):
    import urllib.request

    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False

def urlIteration(driver,urls,post_id,comment_id):

    import time
    from tqdm import tqdm
    from datetime import datetime
    from random import randint

    left_urls = list(urls['URL'])
    time.sleep(randint(5,22))

    current_account_set_number = 1
    start_minute = datetime.now().minute

    for url in tqdm(left_urls):

        post_id = post_id + 1

        driver.get(url)

        try:

            company = urls[urls['URL'] == url]['company'].iloc[0]
            time.sleep(randint(5,12))

            try:
                post_text = scrapePost(driver,post_id, company)

                if any(string in post_text.lower() for string in skip_strs):

                    continue

                else:

                    loadComments(driver)
                    scrapeComments(driver, comment_id,post_id, company)
                    scraping_frame.at[scraping_frame[scraping_frame['URL'] == url].index[0],'done'] = 1

            except:
                driver.close()
                driver, useraccount_used = instagramLogin()


            if post_id % randint(8,12) == 0:
                driver.close()
                driver, useraccount_used = instagramLogin()

        except:

            if connect() == False:

                while connect() == False:
                    time.sleep(120)

            else:
                pass

instagram_sequence, useraccount_old = instagramLogin()
company_list = ['bangolufsen','fractalofficial']

for company in company_list:

    searchCompany(instagram_sequence,company)
    # get unique urls
    urls = fetchPosts(instagram_sequence)
    urls = list(set(urls))
    len(urls)
    pd.DataFrame(urls).to_csv(f'{company}_urls.csv')
    instagram_sequence.close()
    instagram_sequence, useraccount_old = instagramLogin()

scraping_frame = pd.read_csv('scraping_frame_3.csv').drop('Unnamed: 0', axis = 1)

scraping_frame = appendScrapingUrls(scraping_frame, 'fishing_king_official')
scraping_frame.to_csv('scraping_frame_feb.csv')

import time
#time.sleep(3600)

scraping_frame = pd.read_csv('scraping_frame_3.csv', sep = ',').drop('Unnamed: 0', axis = 1)

for scraping_batch in range(30):

    post_id = pd.read_csv('posts.csv', sep = ';').shape[0] + 1

    try:
        instagram_sequence, useraccount_new = instagramLogin()

        try:
            if useraccount_new[0] == useraccount_old[0]:

                instagram_sequence, useraccount_new = instagramLogin()
            else:
                pass
        except:
            pass

        subscraping_frame = subScrapingFrame(scraping_frame, company_list,randint(5,10))
        urlIteration(instagram_sequence,subscraping_frame,post_id, comment_id)
        instagram_sequence.close()
        useraccount_old = useraccount_new

    except:
        pass

    scraping_frame.to_csv('scraping_frame_3.csv')
