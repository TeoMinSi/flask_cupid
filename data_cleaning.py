import pandas as pd
import numpy as np

#cleaning the cols with Others (please specify) except categories col
def replace_others(df):
    
    #industry col
    for i in df.index:
        val_i = df.at[i,'industry']
        replacement_i = df.at[i,'industry_10_TEXT']
        if val_i == 'Others (Please Specify)':
            df.at[i, 'industry'] = val_i.replace("Others (Please Specify)", replacement_i).rstrip()
    
    #payment_method col
    for j in df.index:
        val_j = df.at[j,'payment_method']
        replacement_j = df.at[j,'payment_method_5_TEXT']
        if 'Others (Please Specify)' in val_j:
            df.at[j,'payment_method'] = val_j.replace("Others (Please Specify)",replacement_j).rstrip()

    #card_ownership col        
    for k in df.index:
        val_k = df.at[k,'card_ownership']
        replacement_k = df.at[k,'card_ownership_6_TEXT']
        replacement_k = str(replacement_k)
        #note for standard chartered - remove Bank
        if 'Bank' in replacement_k.title():
            replacement_k = replacement_k.title().replace("Bank", "").rstrip()
        
        #capitalise standard chartered
        if 'Standard Chartered' in replacement_k.title():
            replacement_k = replacement_k.title().rstrip()
            
        if 'Others (Please Specify)' in val_k:
            df.at[k,'card_ownership'] = val_k.replace("Others (Please Specify)",replacement_k.title()).rstrip().rstrip()
    return df

#focus on selected categories
def remove_categories(df1):
    #restaurant, fffc, beverage, gorcery, electronic, sports, clothing, household, hba
    selected_merchant = {'Restaurant (Includes Takeaways)':'restaurant',
                        'Fast Food / Food Court (Includes Takeaways)': 'fffc',
                        'Groceries':'grocery',
                        'Electronic Product':'electronic',
                        'Sports / Outdoor Equipment':'sports',
                        'Clothing':'clothing',
                        'Household Products':'household',
                        'Health & Beauty Products':'hba',
                        'Bubble Tea / Beverages':'beverage'}
    for l in df1.index:
        new_str = ""
        val_l = df1.at[l,'categories']
        li = list(val_l.split(","))
        for m in range(len(li)):
            if li[m] in selected_merchant:
                    new_str += (selected_merchant[li[m]]) +","
        if new_str[len(new_str)-1] == ",":
            new_str = new_str[:-1]

        df1.at[l,'categories'] = new_str
    return df1

#remove whitespaces used in set_range for filter_m fn
def strip_whitespace(val_space):
    val_space = str(val_space).replace(" ","")
    return val_space

#drop all non-numeric characters after checking for ranges used in set_range and filter_m fn
def filter_nonnumeric(val_nonnumeric):
    numeric_filter = filter(str.isnumeric,val_nonnumeric)
    numeric_string = "".join(numeric_filter)
    if numeric_string =="":
        return 0
    else:
        numeric_float = float(numeric_string)
    return numeric_float

#for range calculation used in set_range for filter_m fn
def cal_range(lower_end, upper_end):
    if lower_end=="" and upper_end=="":
        avg_m = 0
    elif lower_end=="":
         avg_m = float(upper_end)
    elif upper_end=="" or upper_end==0:
        avg_m = float(lower_end)
    else:
        avg_m = (float(lower_end) + float(upper_end))/2
    return avg_m

#check for range used for filer_m fn
def set_range(val_range, sym):
    val_range = strip_whitespace(val_range)
    if sym=="to":
        to_index = val_range.find(sym)
        lower_end = val_range [:to_index]
        lower_end = filter_nonnumeric(lower_end)
        upper_end = val_range [to_index+2:]
        upper_end = filter_nonnumeric(upper_end)
        
    else:
        to_index = val_range.find(sym)
        lower_end = val_range [:to_index]
        lower_end = filter_nonnumeric(lower_end)
        upper_end = val_range [to_index+1:]
        upper_end = filter_nonnumeric(upper_end)    

    val_range = cal_range(lower_end, upper_end)
    return val_range

#for category_average
def filter_m(df2):
    for m in range(16, len(df2.columns),3):
        for n in df2.index:
            val_m = df2[df2.columns[m]][n]

            #numeric input 
            if str(val_m).isnumeric():
                replacement_m = float(val_m)

            #check for null input
            elif str(val_m).isalpha() or val_m == "" or str(val_m).lower() =="nil" or val_m=="-" or pd.isna(val_m):
                    replacement_m = 0.0
            
            else:
                #check for range of value
                symbols = ["-","~",":",";","/","to","."]
                for symbol in symbols:
                    if symbol in val_m: 
                        replacement_m = set_range(val_m,symbol)
                        break
                else:
                    replacement_m = filter_nonnumeric(val_m)
                
                        
            df2[df2.columns[m]][n] = replacement_m
    return df2

#binning the average spending in each category
def bin_avg_spend(df):
    #for restaurants
    bins = [0,100,200,300,400,500]
    labels = ["0 to 100","100 to 200","200 to 300","300 to 400",'400 to 500']
    df["bin_restaurant_avg"] = pd.cut(df["restaurant_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_restaurant_avg'] = df['bin_restaurant_avg'].replace(np.nan, "More than 500")
    #print(df[['restaurant_average', 'bin_rest_avg']])

    #for fffc
    bins = [0,50,100,150,200,250,300]
    labels = ["0 to 50","50 to 100","100 to 150","150 to 200",'200 to 250','250 to 300']
    df["bin_fffc_avg"] = pd.cut(df["fffc_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_fffc_avg'] = df['bin_fffc_avg'].replace(np.nan, "More than 300")
    #print(df[['fffc_average', 'bin_fffc_avg']])

    #for grocery
    bins = [0,100,200,300,400,500]
    labels = ["0 to 100","100 to 200","200 to 300","300 to 400",'400 to 500']
    df["bin_grocery_avg"] = pd.cut(df["grocery_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_grocery_avg'] = df['bin_grocery_avg'].replace(np.nan, "More than 500")

    #for electronics
    bins = [0,1,100]
    labels = ["0","1 to 100"]
    df["bin_electronic_avg"] = pd.cut(df["electronic_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_electronic_avg'] = df['bin_electronic_avg'].replace(np.nan, "More than 100")

    #sports
    bins = [0,1,50,100]
    labels = ["0","1 to 50","50 to 100"]
    df["bin_sports_avg"] = pd.cut(df["sports_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_sports_avg'] = df['bin_sports_avg'].replace(np.nan, "More than 100")

    #clothing
    bins = [0,100,200,300,400,500]
    labels = ["0 to 100","100 to 200","200 to 300","300 to 400",'400 to 500']
    df["bin_clothing_avg"] = pd.cut(df["clothing_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_clothing_avg'] = df['bin_clothing_avg'].replace(np.nan, "More than 500")

    #hba
    bins = [0,1,50,100,150,200,250,300]
    labels = ["0","1 to 50","50 to 100","100 to 150","150 to 200",'200 to 250','250 to 300']
    df["bin_hba_avg"] = pd.cut(df["hba_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_hba_avg'] = df['bin_hba_avg'].replace(np.nan, "More than 300")

    #household
    bins = [0,1,50,100,150,200,250,300]
    labels = ["0","1 to 50","50 to 100","100 to 150","150 to 200",'200 to 250','250 to 300']
    df["bin_household_avg"] = pd.cut(df["household_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_household_avg'] = df['bin_household_avg'].replace(np.nan, "More than 300")

    #beverage
    bins = [0,1,20,40,60,80,100]
    labels = ["0","1 to 20","20 to 40","40 to 60","60 to 80",'80 to 100']
    df["bin_beverage_avg"] = pd.cut(df["beverage_average"], bins = bins, labels = labels, include_lowest = True)
    df['bin_beverage_avg'] = df['bin_beverage_avg'].replace(np.nan, "More than 100")
    
    return df

def data_cleaning(df):
    survey = df
    survey.drop([0,1], inplace=True)

    #drop background cols and remove non-selected merchant cols
    survey.drop(columns = [ 'book','book_average','book_recency',
                            'footwear','footwear_average', 'footwear_recency',
                            'jw', 'jw_average', 'jw_recency',
                            'furniture','furniture_average', 'furniture_recency',
                            'news', 'news_average','news_recency',
                            'toys', 'toys_average', 'toys_recency',
                            'gardening','gardening_average', 'gardening_recency',
                            'others', 'others_average','others_recency'],inplace=True)

    #update survey for cleaning of others (please specify) and removing non-shortlisted categories
    replace_others_clean = replace_others(survey)
    remove_categories_clean = remove_categories(replace_others_clean)
    filter_m_clean = filter_m(remove_categories_clean)

    #drop the others (please specify) col
    filter_m_clean.drop(columns=['industry_10_TEXT','payment_method_5_TEXT',
                                'card_ownership_6_TEXT','categories_17_TEXT'], inplace = True)

    #binning of the average spending in each category
    filter_m_clean = bin_avg_spend(filter_m_clean)

    #insert customer_id col                  
    filter_m_clean.reset_index(inplace=True, drop=True)
    filter_m_clean.insert(0, 'customer_id', filter_m_clean.index)
    return filter_m_clean
