import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def freq_numeric(freq):
    if freq == 'Daily':
        value = 30

    elif freq == '2-3 times per week':
        value = 10
    
    elif freq == '4-6 times per week':
        value = 20

    elif freq == 'Weekly':
        value = 4
    
    elif freq == 'Around 2-3 times a month':
        value = 2
    
    elif freq == 'Monthly':
        value = 1
    
    else:
        value = 0

    return value

def rec_numeric(rec):
    if rec == '< 3 days ago':
        value = 3

    elif rec == '< 1 week ago':
        value = 7
    
    elif rec == '< 2 weeks ago':
        value = 14

    elif rec == '< a month ago':
        value =30
    
    elif rec == '> a month':
        value = 60
    
    else:
        value = 1

    return value

def RFM_segment(RFM):
    if RFM == '444':
        segment = 'Champions'
        
    elif RFM[2] == '4':
        segment = 'Big spenders'  
        
    elif RFM[1] == '4':
        segment = 'Loyal customers'

    elif RFM[0] == '4':
        segment = 'Recent customers'
        
    elif RFM[0] == '1':
        segment = 'Almost lost'
        
    else:
        segment = 'Regulars'
    
    return segment


def rfm_analysis(df):
    df2 =df[['customer_id','age_group', 'gender', 'region', 'industry', 'annual_income', 'education', 'payment_method', 'card_ownership', 'discounts', 'categories']]
    column_names = ['customer_id','recency', 'frequency', 'monetary']
    transactions= pd.DataFrame(columns = column_names)
    transactions['customer_id'] = df['customer_id']
    transactions['frequency'] = df['frequency'].apply(freq_numeric)
    recency_df = df[['restaurant_recency', 'fffc_recency', 'grocery_recency', 'electronic_recency', 'sports_recency', 'clothing_recency', 'hba_recency', 'household_recency', 'beverage_recency']]
    column_names = ['restaurant_recency', 'fffc_recency', 'grocery_recency', 'electronic_recency', 'sports_recency', 'clothing_recency', 'hba_recency', 'household_recency', 'beverage_recency']
    recency_df_updated = pd.DataFrame(columns = column_names)
    length = len(recency_df.columns)

    for i in range(length):
        lst = []
        for j in recency_df.index:
            lst.append(rec_numeric(recency_df[recency_df.columns[i]][j]))
        recency_df_updated[column_names[i]] = lst
    
    transactions['recency'] = recency_df_updated.sum(axis=1)
    monetary_df = df[['restaurant_average', 'fffc_average', 'grocery_average', 'electronic_average', 'sports_average', 'clothing_average', 'hba_average', 'household_average', 'beverage_average']]
    transactions['monetary'] = monetary_df[['restaurant_average', 'fffc_average', 'grocery_average', 'electronic_average', 'sports_average', 'clothing_average', 'hba_average', 'household_average', 'beverage_average']].sum(axis=1)
    transactions = transactions[transactions['monetary'] != 0]
    rfm = pd.DataFrame()
    rfm['frequency'] = transactions['frequency']
    rfm['log_monetary'] = round(np.log(transactions['monetary']),2)
    rfm['log_recency'] = round(np.log(transactions['recency']),2)
    scaler = StandardScaler()
    scaler.fit(rfm)
    rfm_prepared = scaler.transform(rfm)
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)

    # Assign these labels to 5 equal percentile groups 
    r_groups = pd.qcut(rfm['log_recency'], q=4, labels=r_labels)
    f_groups = pd.qcut(rfm['frequency'], q=5, labels=f_labels, duplicates='drop')
    m_groups = pd.qcut(rfm['log_monetary'], q=4, labels=m_labels)

    # Create new columns R, F and M
    rfm = rfm.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)
    def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
    rfm['RFM'] = rfm.apply(join_rfm, axis=1)
    transactions_c = transactions.copy()
    transactions_c['RFM'] = rfm['RFM']
    transactions_c['RFM_segment'] = transactions_c['RFM'].apply(RFM_segment)
    df2_c = df2.copy()
    df2_c['RFM'] = transactions_c['RFM']
    df2_c['RFM_segment'] = transactions_c['RFM_segment']
    df2_c['average_recency'] = transactions_c['recency']
    df2_c['average_frequency'] = transactions_c['frequency']
    df2_c['average_monetary'] = transactions_c['monetary']

    df2_c.dropna(inplace = True)
    #df2_c.to_csv('011120_cust_db.csv')
    return df2_c

# df = pd.read_csv('22092020 survey_cleaned.csv')
# df = rfm_analysis(df)
# df.to_csv('221020_cust_db.csv')



