import re
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

def encode_data(datapoint):
    if datapoint <= 0:
        return 0
    if datapoint >= 1:
        return 1

def mba(survey):
    #selecting customer_id and merchant cols
    results = survey[["customer_id", "restaurant", "fffc", "grocery", "electronic", "sports", "clothing", "household", "hba", "beverage"]]

    #creating a new df with a col of combination of all merchants in a str (done cause of chained assignnment)
    merchant = pd.DataFrame()
    merchant["merchants"] = results[results.columns[1:]].apply(lambda y:','.join(y.dropna().astype(str)),axis=1)

    #create another df with just the combined merchants in str and customer_id
    new_merchant = pd.concat([results["customer_id"], merchant["merchants"]], axis = 1)

    #text pre-processing
    #converting all inputs to lowercase
    new_merchant = new_merchant.applymap(lambda s:s.lower() if type(s) == str else s)

    #replace str containing 'and' with ','
    new_merchant = new_merchant.applymap(lambda x :str(x).replace('and ', ','))

    #strip data of all whitespaces, not done here to match merchants_db input
    #new_merchant = new_merchant.applymap(lambda r: str(r).replace(" ",""))
    #new_merchant = new_merchant.applymap(lambda w: str(w).strip())

    #change merchants col from str to list
    new_merchant["merchants"] = new_merchant["merchants"].str.split('[:;.,/]')

    #explode merchants cols and ensure index reset since customer_id is the identifier and already in col
    exploded_merchant = new_merchant.explode("merchants")
    exploded_merchant.reset_index(drop=True , inplace=True)

    #check for nil or - values or empty string values (survey input error)
    for j in exploded_merchant.index:
        if exploded_merchant["merchants"][j] == "nil" or exploded_merchant["merchants"][j] =="Nil" or exploded_merchant["merchants"][j]=="" or exploded_merchant["merchants"][j]=="nan" or exploded_merchant["merchants"][j]=="-":
            exploded_merchant.drop([j], inplace=True)

    #remove front and trailing whitespaces 
    exploded_merchant = exploded_merchant.applymap(lambda w: str(w).strip())

    #for mba add a quantity col
    exploded_merchant.insert(2, "quantity", 1)

    #explodexploded_merchantcsv', index = False)
    #print(exploded_merchant)
    #note exploded_merchantls are in str and quantity col in numpy.float

    market_basket = exploded_merchant.groupby(['customer_id', 'merchants'])['quantity']
    market_basket = market_basket.sum().unstack().reset_index().fillna(0).set_index('customer_id')
    market_basket = market_basket.applymap(encode_data)

    #itemsets are possible generation after applying apriori on min_support
    itemsets = apriori(market_basket, min_support=0.07, use_colnames=True)

    #rules is a df of possible associations
    rules = association_rules(itemsets, metric="lift", min_threshold=0)

    #note sorting done after appending categories or else error

    #check fronzen set antecedents for single item (sort by category)
    for k in rules.index:
        if len(rules["antecedents"][k]) > 1:
            rules.drop([k], inplace=True)

    #convert fronzenset from itemset into list or string(for single item)
    rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
    rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

    #round values to 3dp for visualization
    rules['antecedent support'] = round(rules['antecedent support'],3)
    rules['consequent support'] = round(rules['consequent support'],3)
    rules['support'] = round(rules['support'],3)
    rules['confidence'] = round(rules['confidence'],3)
    rules['lift'] = round(rules['lift'],3)
    rules['leverage'] = round(rules['leverage'],3)
    rules['conviction'] = round(rules['conviction'],3)

    #creating new cols for graphs 
    #new col = itemset
    rules['itemsets'] = rules['antecedents']+' -> '+ rules['consequents']

    #new col = count, assign int instead of float
    rules['count'] = round(rules["support"] * len(new_merchant.index))
    rules["count"] = rules["count"].astype(int)

    #rules.to_csv("mba_007.csv")
    return rules

#function that helps to create a col category based on antecedents
def concat_categories(survey,df_categories):
    df_rules = mba(survey)
    antecedents_category = []
    #antecedents_array = df_rules["antecedents"].to_numpy()
    for i in range(df_rules.index[-1]+1):
        for j in range(len(df_categories.index)):
            #note use of try and except cause apriori will lead to missing index (can consider resetting index)
            try:
                if df_rules["antecedents"][i] == df_categories["Merchant"][j]:
                    antecedents_category.append(df_categories["Category"][j])
                    break
            except KeyError:
                continue
    df_rules['antecedents_category'] = antecedents_category
    #sort rules_df by lift
    df_rules_sort = df_rules.sort_values(by=['lift'], ascending=False)
    return df_rules_sort

# rules_w_cat = (concat_categories(categories))
# print(rules_w_cat)

def top_merchant(input_category):
    input_df = concat_categories(categories)
    selected_categorylist_index = input_df.index[input_df["antecedents_category"]==input_category].tolist()
    selected_category_index = selected_categorylist_index[0]
    selected_antecedent = input_df["antecedents"][selected_category_index]
    selected_consequent = input_df["consequents"][selected_category_index]
    return "The top suggested merchant for this category is " + selected_antecedent + " and people who buy from this merchant also shop at " + selected_consequent +"."
