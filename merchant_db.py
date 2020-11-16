import pandas as pd
import numpy as np
import os
import re
from twython import Twython
import json
import math
import nltk
import os.path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from sqlalchemy import create_engine, types

engine = create_engine('postgresql://minsi2:password@localhost:5432/fyp_database')

def merchant_cleaning(df):
    #Selecting only merchants columns
    df2 = df[["restaurant", "fffc", "grocery", "electronic", "sports", "clothing", "hba",  "household", "beverage"]]

    #Converting all data inputs to lowercase 
    df2 = df2.applymap(lambda s:s.lower() if type(s) == str else s)

    #For exception of our survey inputs,
    #Replace str that contains 'and' with ','
    df2 = df2.applymap(lambda x :str(x).replace('and ', ','))

    #Convert str output to list of merchants via multiple delimiters
    df2 = df2.applymap(lambda y :re.split('[:;.,/]', str(y)))

    length = len(df2.columns)
    categories = list(df2.columns)

    merchants_db = pd.DataFrame(columns=['Category', 'Merchant', 'Freq'])

    for i in range(length):
        
        #Splitting list of merchants to unique row
        new_df = df2.iloc[:, i].explode()
        
        #removing space in front/ end of merchant name
        new_df = new_df.str.strip()
        
        
        #Create new Merchant dataset
        data = [new_df]
        headers = ["Merchant"]
        merchants = pd.concat(data, axis=1, keys=headers)
        

        #Inserting new column  of category name
        merchants.insert(0, 'Category', categories[i])
        
        
        #Creating a column for frequency counts of each merchant
        merchants['Freq'] = merchants.groupby(['Merchant']).transform('count')

        
        #Sort merchants count via descending order
        merchants = merchants.sort_values(by ='Freq', ascending = False)
        
        #Drop rows with NAN values
        merchants = merchants[merchants.Merchant != 'nan']
        
        #Drop rows with 'none' values
        merchants = merchants[merchants.Merchant != 'none']
        
        #For the exception of our survey
        #Removing exceptions where merchant name = 'nil' 
        merchants = merchants[merchants.Merchant != 'nil']
        
        #Removing exceptions where merchant name = '-'
        merchants = merchants[merchants.Merchant != '-']
        
        #Removing merchants with merchant name = 'online'
        merchants = merchants[merchants.Merchant != 'online']
        
        #Removing merchants with merchant name = ''
        merchants = merchants[merchants.Merchant != '']
        
        #Drop duplicated rows of merchants
        merchants = merchants.drop_duplicates(subset= 'Merchant', keep='first', inplace=False)
        
        merchants_db = merchants_db.append(merchants, ignore_index=True)
        
    #Creating new column for auto-increment Merchant ID
    merchants_db.insert(0, 'MerchantID', merchants_db.index)

    #Finalised merchant database csv
    merchants_db.to_csv('16102020_merchants_db.csv', index= False)
    
    return merchants_db

#Adding score categories
def sentiment_category(score):
    if score > 70:
        category = 'Very Positive'
    elif score > 20:
        category = 'Positive'
    elif score > 0:
        category = 'Slightly Positive'
    elif score == 0:
        category = 'Neutral'
    elif score > -20:
        category ='Slightly Negative'
    elif score > -50:
        category = 'Negative'
    else:
        category = 'Very Negative'
    return category

def sentiment_analysis(df):
    category_list = ["restaurant", "fffc", "grocery", "electronic", "sports", "clothing", "hba", "household", "beverage"]

    merchants_compiled = pd.DataFrame(columns=['MerchantID','Category', 'Merchant', 'Freq'])

    for category in category_list:
        
        #Creating data subsets of different categories
        merchants = df[df["Category"] == category]
        
        #No. of merchants in each category
        num = merchants['Merchant'].count()
        
        #scrap the top 30% merchants in each category, if there's decimals, round up to the next whole int
        merchants_scrapped = merchants.iloc[0:math.ceil(num*0.30)]

        #Output list of top 30% merchants (to be scrapped) into new dataframe
        merchants_compiled = merchants_compiled.append(merchants_scrapped, ignore_index=True)
    
    #List of merchants 
    merchants_list = merchants_compiled["Merchant"]

    #list to store all merchants scores
    merchants_scores_list = []

    #list to store all merchants comments
    merchant_comment_lst = []

    merchants = list(merchants_compiled['Merchant'])
    path = os.path.abspath('twitter_credentials.json')
    
    # Load twitter credentials from json file
    with open(path,'r') as file:
        creds = json.load(file)

        
    # Instantiate an object
    python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])


    # Create our query
    for i in range(len(merchants_list)):
        query = {'q': merchants_list[i],
                'result_type': 'recent',
                #specify singaporean twitter users only
                'geocode':['1.290270','103.851959','10km'],
                #100 comments max, 7-days limit
                'count': 100,
                'lang': 'en',
                }

        dict_ = {
            'user': [],
            'date': [],
            'text': [],
            'favorite_count': []
            }


        for status in python_tweets.search(**query)['statuses']:
            dict_['user'].append(status['user']['screen_name'])
            dict_['date'].append(status['created_at'])
            dict_['text'].append(status['text'])
            dict_['favorite_count'].append(status['favorite_count'])

        merchants_list[i] = pd.DataFrame(dict_)

        
        #Converting twitter usernames into lowercases
        merchants_list[i]['user'] = merchants_list[i]['user'].astype(str).str.lower()
        
        #Removing users with usernames containing merchant names
        merchants_list[i] = merchants_list[i][~merchants_list[i]['user'].astype(str).str.contains('merchants_list[i]')]
        
        
        #Sentiment analysis for each scraped merchant dataframe
        sid = SentimentIntensityAnalyzer()

        #Adding sentiments scores column to existing dataframe
        merchants_list[i]['scores'] = merchants_list[i]['text'].apply(lambda text: sid.polarity_scores(text))
        
        # Retrieving compounded sentiment scores
        merchants_list[i]['compound']  = merchants_list[i]['scores'].apply(lambda score_dict: score_dict['compound'])
        
        #adding a Positive / Negative sentiment column
        merchants_list[i]['sentiment'] = merchants_list[i]['compound'].apply(lambda compound: 'pos' if compound >=0 else 'neg')

        sentiment_score = 0
        
        no_of_reviews = merchants_list[i]['user'].count()

        compound = merchants_list[i]['compound']

        favorite_count = merchants_list[i]['favorite_count']
        
        #overall sentiment score will be amplified by no. of favorites for the particular tweet
        for j in range(len(compound)):
            try:
                overall_score = compound[j]*(favorite_count[j]+1)
            except:
                KeyError
            sentiment_score += overall_score

        merchant_score = (round((sentiment_score*100) /no_of_reviews,2))
        merchants_scores_list.append(merchant_score)
        
        #retrieving compilation of comments for each merchant
        
        comment_str = ""
        updated_comment_str = ""
        comment = merchants_list[i]['text']
        comment = comment.astype(str).str.lower()
        for i in range(len(comment)):
            comment_str += comment[i] + " "
        try:
            comment_str = re.sub('[^0-9a-zA-Z @]+', '', comment_str)
            comment_str = comment_str.replace("rt","")
            comment_lst = comment_str.split()
            #remove weird hyperlinks
            for word in comment_lst:
                if 'http' not in word and 'https' not in word and 'com' not in word and 'www' not in word and '@' not in word:
                    updated_comment_str += word + " "
        except:
            comment.empty
            KeyError
        merchant_comment_lst.append(updated_comment_str)
    
    #Appending merchants scores to original df
    merchants_compiled['score'] = merchants_scores_list
    merchants_compiled['comments'] = merchant_comment_lst

    #if score==NaN, replace with 0
    merchants_compiled = merchants_compiled.fillna(0)

    merchants_compiled['score_category'] = merchants_compiled['score'].apply(sentiment_category)

    merchants_compiled.drop('Merchant', axis=1, inplace=True)

    merchants_compiled.insert(2, 'Merchant', merchants)
    #merchants_compiled.to_csv('16102020_merchant_score.csv')
    return merchants_compiled


# df = pd.read_sql_query('SELECT * FROM clean_raw_data;',engine)
# final_merchant_df = merchant_cleaning(df)
# final_merchant_df.to_csv('merchant_score_check.csv')

