import pandas as pd
import numpy as np
import random
import csv
import json
import warnings
from operator import itemgetter

warnings.filterwarnings('ignore')

#library imports for machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


def mba_recommendation(mba_db, score_db, pre_cat):
    #score_db rows with same category sorted by score descending
    cat_df = score_db[score_db["Category"]==pre_cat]
    cat_df_c = cat_df.copy()
    cat_df_c.sort_values(by = ['score'], ascending=False, inplace=True)
    cat_df_c.reset_index(inplace=True, drop=True)

    #mba_db rows with same category (already sorted) and list the antecedents
    mba_cat_df = mba_db[mba_db["antecedents_category"]==pre_cat]
    mba_cat_df_c = mba_cat_df.copy()
    mba_lst = mba_cat_df['antecedents'].to_list()
    
    #if top merchant in mba return
    for i in cat_df_c.index:
        merchant = cat_df_c['Merchant'][i]
        if merchant in mba_lst:
            possible_assc_df = mba_cat_df_c[mba_cat_df_c["antecedents"]==merchant]
            possible_assc_df.reset_index(inplace=True, drop=True)
            return possible_assc_df["antecedents"][0] + " -> " + possible_assc_df["consequents"][0]

    #else return top merchant with None
    return cat_df_c['Merchant'][0] + " -> " + 'None'

# mba_db = pd.read_sql_query('SELECT * FROM mba;',engine)
# score_db = pd.read_sql_query('SELECT * FROM merchant_score;',engine)
# print(mba_recommendation(mba_db, score_db, 'grocery'))

def convert_dict(form_dict):
    #print(form_dict)

    gender, age, annual_income, education, region, industry = itemgetter('gender', 'age', 'annual_income', 'education', 'region','industry')(form_dict)

    #list to split input
    age_list=['< 20 years old', '21-30 years old',  '31-40 years old', '41-50 years old', '> 50 years old']
    income_list=['I am a Student', '< $20,000','$20,001 - $44,999', '$45,000 - $59,999','$60,000 - $89,999','$90,000 - $120,000','> $120,000']
    edu_list=['Primary School', 'O Levels','A Levels', 'Diploma','University Degree','Masters Degree','PhD']
    region_list=["Central/South","East","North","North-East","West"]
    ind_list=["Business/HR","Construction","Consulting","Engineering","Events","Finance","Floral & Landscaping",
            "Housewife","Humanities","Law","Life Sciences","Mathematics","Media","Medical","Oil and gas",
            "Publishing","Real Estate and Fengshui","Retail","Sales","Security","Services","Social Sciences",
            "Technology","Tourism/Hospitality","Travel","Wedding"]

    form=[]

    #split gender
    if gender=='Male':
        form.append(1)
        #rint(form)
    else:
        form.append(0)
        # print(form)
        
    #split age
    for i in range(len(age_list)):
        if age==age_list[i]:
            form.append(i)
            # print(form)

    #split income
    for i in range(len(income_list)):
        if annual_income==income_list[i]:
            form.append(i)
            # print(form)
            
    #split education
    for i in range(len(edu_list)):
        if education==edu_list[i]:
            form.append(i)
            # print(form)
    
    # print(form)

    #split region
    for i in region_list:
        if region==i:
            form.append(1)
            # print(form)
        else:
            form.append(0)
            # print(form)
    
    # print(form)
            
    #split industry
    for i in ind_list:
        if industry==i:
            form.append(1)
            # print(form)
        else:
            form.append(0)
            # print(form)
    
    
    return form


#df = clean_raw_data.csv
def cat_prediction(df, form):

    #output list of predicted categories
    cat_of_int=[]
    
    # data preprocessing 
    df['restaurant'] = pd.Series([0 for x in range(len(df.index))])
    df['fffc'] = pd.Series([0 for x in range(len(df.index))])
    df['beverage'] = pd.Series([0 for x in range(len(df.index))])
    df['grocery'] = pd.Series([0 for x in range(len(df.index))])
    df['electronic'] = pd.Series([0 for x in range(len(df.index))])
    df['sports'] = pd.Series([0 for x in range(len(df.index))])
    df['clothing'] = pd.Series([0 for x in range(len(df.index))])
    df['household'] = pd.Series([0 for x in range(len(df.index))])
    df['hba'] = pd.Series([0 for x in range(len(df.index))])

    for i in range (0,len(df)):
        row= df['categories'].values[i].split(',')
        for j in row:
            if j =='restaurant':
                df['restaurant'][i]=1
            elif j =='fffc':
                df['fffc'][i]=1
            elif j =='beverage':
                df['beverage'][i]=1
            elif j =='grocery':
                df['grocery'][i]=1
            elif j =='electronic':
                df['electronic'][i]=1
            elif j =='sports':
                df['sports'][i]=1
            elif j =='clothing':
                df['clothing'][i]=1
            elif j =='household':
                df['household'][i]=1
            elif j =='hba':
                df['hba'][i]=1

    dropped_df = df.drop(['customer_id','payment_method','card_ownership','discounts','frequency','categories',
        'restaurant_average','restaurant_recency','fffc_average','fffc_recency', 'grocery_average',
        'grocery_recency', 'electronic_average', 'electronic_recency', 'sports_average', 
        'sports_recency','clothing_average', 'clothing_recency', 'hba_average', 'hba_recency',
        'household_average', 'household_recency','beverage_average', 'beverage_recency',
        'bin_restaurant_avg','bin_fffc_avg', 'bin_grocery_avg', 'bin_electronic_avg',
        'bin_sports_avg', 'bin_clothing_avg', 'bin_hba_avg','bin_household_avg', 'bin_beverage_avg'], axis=1)
    
    #binary encoding for gender column (male = 1, female = 0)
    dropped_df['Gender'] = dropped_df['gender'].map( {'Male':1, 'Female':0})

    #numeric encoding for columns with order
    # age group
    dropped_df['agegroup_encoded'] = dropped_df['age_group'].map( {'< 20 years old':0, '21-30 years old':1, 
                                                                '31-40 years old':2, '41-50 years old':3,
                                                                '> 50 years old':4})
    # annual income
    dropped_df['income_encoded'] = dropped_df['annual_income'].map( {'I am a Student':0, '< $20,000':1, 
                                                                '$20,001 - $44,999':2, '$45,000 - $59,999':3,
                                                                '$60,000 - $89,999':4,'$90,000 - $120,000':5,
                                                                    '> $120,000':6})
    #education
    dropped_df['education_encoded'] = df['education'].map( {'Primary School':0, 'O Levels':1, 
                                                'A Levels':2, 'Diploma':3,
                                                'University Degree':4,'Masters Degree':5,'PhD':6})

    # one hot encoding for columns with no order
    # region
    dropped_df = pd.concat([dropped_df, pd.get_dummies(dropped_df['region'], prefix='region')], axis=1)

    #industry
    dropped_df = pd.concat([dropped_df, pd.get_dummies(dropped_df['industry'], prefix='industry')], axis=1)

    #drop all the non-encoded columns
    dropped_df = dropped_df.drop(['age_group','gender','region','industry','annual_income','education'],axis=1)

    #separate all the output columns into different dataframes
    restaurant_df = dropped_df.drop(['fffc','grocery','electronic','sports','clothing','hba','household','beverage',],axis=1)
    clothing_df = dropped_df.drop(['restaurant','fffc','grocery','electronic','sports','hba','household','beverage',],axis=1)
    beverage_df = dropped_df.drop(['restaurant','fffc','grocery','electronic','sports','clothing','hba','household',],axis=1)
    fffc_df = dropped_df.drop(['restaurant','grocery','electronic','sports','clothing','hba','household','beverage',],axis=1)
    grocery_df = dropped_df.drop(['restaurant','fffc','electronic','sports','clothing','hba','household','beverage',],axis=1)

    #restaurant dataframe
    #preparing dataframe for training
    X = restaurant_df.drop(['restaurant'],axis=1)
    y = restaurant_df['restaurant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

    #random forest model
    rfc = RandomForestClassifier(bootstrap= True, ccp_alpha= 0.0, class_weight= None, criterion= 'gini', max_depth= None, max_features=  'auto', max_leaf_nodes= None, max_samples= None, min_impurity_decrease= 0.0, min_impurity_split= None, min_samples_leaf= 1, min_samples_split= 2, min_weight_fraction_leaf= 0.0, n_estimators= 100, n_jobs= None, oob_score= False, random_state= 67, verbose= 0, warm_start= False)

    rfc.fit(X_train,y_train)
    print(X_train.columns)
    X = [form]
    print(X)
    test_predict = rfc.predict(X)
    if test_predict==[1]:
        cat_of_int.append("restaurant")

    #clothing dataframe
    #preparing dataframe for training
    X = clothing_df.drop(['clothing'],axis=1)
    y = clothing_df['clothing']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

    #random forest model
    rfc = RandomForestClassifier(bootstrap= True, ccp_alpha= 0.0, class_weight= None, criterion= 'gini', max_depth= None, max_features=  'auto', max_leaf_nodes= None, max_samples= None, min_impurity_decrease= 0.0, min_impurity_split= None, min_samples_leaf= 1, min_samples_split= 2, min_weight_fraction_leaf= 0.0, n_estimators= 100, n_jobs= None, oob_score= False, random_state= 67, verbose= 0, warm_start= False)
    rfc.fit(X_train,y_train)

    X = [form]
    test_predict = rfc.predict(X)
    if test_predict==[1]:
        cat_of_int.append("clothing")

    #beverage dataframe
    #preparing dataframe for training
    X = beverage_df.drop(['beverage'],axis=1)
    y = beverage_df['beverage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #random forest model
    rfc = RandomForestClassifier(bootstrap= True, ccp_alpha= 0.0, class_weight= None, criterion= 'gini', max_depth= None, max_features=  'auto', max_leaf_nodes= None, max_samples= None, min_impurity_decrease= 0.0, min_impurity_split= None, min_samples_leaf= 1, min_samples_split= 2, min_weight_fraction_leaf= 0.0, n_estimators= 100, n_jobs= None, oob_score= False, random_state= 0, verbose= 0, warm_start= False)
    rfc.fit(X_train,y_train)

    X = [form]
    test_predict = rfc.predict(X)
    if test_predict==[1]:
        cat_of_int.append("beverage")

    #grocery dataframe
    #preparing dataframe for training
    X = grocery_df.drop(['grocery'],axis=1)
    y = grocery_df['grocery']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66)

    #random forest model
    rfc = RandomForestClassifier(random_state= 66)
    rfc.fit(X_train,y_train)
    # predictions

    X = [form]
    test_predict = rfc.predict(X)
    if test_predict==[1]:
        cat_of_int.append("grocery")

    #FFFC Dataframe
    X = fffc_df.drop(['fffc'],axis=1)
    y = fffc_df['fffc']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=60)
    rfc = RandomForestClassifier(random_state = 60)
    rfc.fit(X_train,y_train)
    X = [form]
    test_predict = rfc.predict(X)
    if test_predict==[1]:
        cat_of_int.append("fffc")

    return cat_of_int

def rfm_categories(x): 
    if x == [5]:
        return 'Champions'
    elif x==[4]:
        return 'Big spenders'
    elif x==[3]:
        return 'Loyal customers'
    elif x==[2]:
        return 'Recent customers'
    elif x==[1]:
        return 'Regulars'
    else:
        return 'Almost lost'

def rfm_classification(df, form):
    df = df.drop(['customer_id','payment_method','card_ownership',
                    'discounts','categories','RFM','average_recency','average_frequency','average_monetary'], axis=1)
    # one hot encoding for columns with no order
    # numeric encoding for columns with order
    # binary encoding for columns with 2 categories

    #binary encoding for gender column (male = 1, female = 0)
    df['gender_encoded'] = df['gender'].map( {'Male':1, 'Female':0})

    #numeric encoding for columns with order
    # age group
    df['agegroup_encoded'] = df['age_group'].map( {'< 20 years old':0, '21-30 years old':1, 
                                                    '31-40 years old':2, '41-50 years old':3,
                                                    '> 50 years old':4})
    # annual income
    df['income_encoded'] = df['annual_income'].map( {'I am a Student':0, '< $20,000':1, 
                                                '$20,001 - $44,999':2, '$45,000 - $59,999':3,
                                                '$60,000 - $89,999':4,'$90,000 - $120,000':5,
                                                    '> $120,000':6})
    #education
    df['education_encoded'] = df['education'].map( {'Primary School':0, 'O Levels':1, 
                                                'A Levels':2, 'Diploma':3,
                                                'University Degree':4,'Masters Degree':5, 'PhD':6})

    #RFM segment
    df['RFM_segment_encoded'] = df['RFM_segment'].map( {'Almost lost':0, 'Regulars':1, 
                                                'Recent customers':2, 'Loyal customers':3,
                                                'Big spenders':4,'Champions':5})

    # one hot encoding for columns with no order
    # region
    df = pd.concat([df, pd.get_dummies(df['region'], prefix='region')], axis=1)

    #industry
    df = pd.concat([df, pd.get_dummies(df['industry'], prefix='industry')], axis=1)

    #drop all the non-encoded columns
    df = df.drop(['age_group','gender','region','industry','annual_income','education','RFM_segment'],axis=1)

    X = df.drop(['RFM_segment_encoded'],axis=1)
    y = df['RFM_segment_encoded']
    sm = SMOTE(random_state = 2)
    X_res, y_res = sm.fit_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=42)
    dt = DecisionTreeClassifier(criterion = 'entropy', max_depth =22, random_state = 42)
    dt.fit(X_train, y_train) 
    y_predict = dt.predict(X_test)
    X_form = [form]
    dt_predict = dt.predict(X_form)
    print(dt_predict)
    segment = rfm_categories(dt_predict)
    return segment

# from sqlalchemy import create_engine, types
# engine = create_engine('postgresql://minsi2:password@localhost:5432/fyp_database') # enter your password and database names here


# df = pd.read_sql_query('SELECT * FROM clean_raw_data;',engine)
# mba_db = pd.read_sql_query('SELECT * FROM mba;',engine)
# score_db = pd.read_sql_query('SELECT * FROM merchant_score;',engine)
# cust_db = pd.read_sql_query('SELECT * FROM customer_datatable;',engine)

# form_dict = {'gender':'Female',
#     'age': '31-40 years old',
#     'annual_income': '$60,000 - $89,999',
#     'education': 'O Levels',
#     'region': 'East',
#     'industry': 'Technology'}

# form = convert_dict(form_dict)

# prediction_list = cat_prediction(df, form)
# segment = rfm_classification(cust_db, form)
# print(form)
# print(prediction_list)
# print(segment)
