import pandas as pd

def allowed_file(filename):
    if '.' not in filename:
        print("Please enter valid file")
        return False
    ext = filename.rsplit('.',1)[1]
    if ext.upper() != "CSV":
        print("Please enter valid file format")
        return False
    return True

def csv_format(df):
    list_of_columns = ['age_group', 'gender', 'region',	'industry', 'industry_10_TEXT', 'annual_income', 'education',	
                        'payment_method', 'payment_method_5_TEXT', 'card_ownership', 'card_ownership_6_TEXT',	
                        'discounts'	, 'frequency', 'categories', 'categories_17_TEXT', 'restaurant', 'restaurant_average', 'restaurant_recency',	
                        'fffc', 'fffc_average', 'fffc_recency',
                        'grocery', 'grocery_average', 'grocery_recency'	,
                        'electronic', 'electronic_average', 'electronic_recency',
                        'book', 'book_average', 'book_recency',
                        'sports', 'sports_average', 'sports_recency',
                        'clothing', 'clothing_average', 'clothing_recency',
                        'footwear', 'footwear_average', 'footwear_recency',
                        'hba', 'hba_average', 'hba_recency',
                        'jw', 'jw_average', 'jw_recency',	
                        'household', 'household_average', 'household_recency',	
                        'furniture', 'furniture_average', 'furniture_recency',
                        'beverage', 'beverage_average', 'beverage_recency',	
                        'news', 'news_average', 'news_recency',
                        'toys', 'toys_average', 'toys_recency',
                        'gardening', 'gardening_average', 'gardening_recency',	
                        'others', 'others_average', 'others_recency'
                        ]
    csv_columns = list(df.columns)
    
    if csv_columns != list_of_columns:
        return False
    
    return True