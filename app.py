#this file holds the flask application
#import libaries/modules


# import requests
from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

import csv
import pandas as pd
from sqlalchemy import create_engine, types
import json
import numpy as np


#import all the local functions
import data_cleaning
import cust_db
import mba_db
import merchant_db
import ML_form
import csv_exceptions

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

#initialise flask app
# ms flask app running on http://127.0.0.1:5000/
app = Flask(__name__)
app.debug = True
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


#database configurations (change it base on your own settings)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://felicia:password@localhost:5432/fyp_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
engine = create_engine('postgresql://felicia:password@localhost:5432/fyp_database') # enter your password and database names here


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

@app.route('/')
def index():
    return render_template('inputData.html')


@app.route('/uploaddata', methods=['POST']) # binds URL to view function
def uploaddata():
    if request.method == 'POST':
        #gets csv file from the form
        csvfile = request.files['csvfile']
        print(csvfile)
        if not csv_exceptions.allowed_file(csvfile.filename):
            # return "Please upload a valid file format"
            return redirect('http://localhost:3000/flask/templates/inputData.html',code = 500)
        
        df = pd.read_csv(csvfile)
        
        if not csv_exceptions.csv_format(df):
            print("Please upload csv file in the right column format")
            return redirect('http://localhost:3000/flask/templates/inputData.html',code = 500)

        # RAW_SURVEY_DATA
        df.to_sql('raw_survey_data',con=engine,index=False,if_exists='replace')
        print ('raw csv successfully uploaded into database')
        print("cleaning raw data")

        #CLEAN_RAW_DATA
        raw_df = pd.read_sql_query('select * from raw_survey_data',con=engine)
        cleaned_df = data_cleaning.data_cleaning(raw_df)
        cleaned_df.to_sql('clean_raw_data',con=engine,index=False,if_exists='replace')   
        print ("successfully uploaded cleaned data into data table")

        #CUSTOMER_DB
        print("extracting and cleaning customer data table")
        cleaned_df = pd.read_sql_query('select * from clean_raw_data',con=engine)
        cust_df = cust_db.rfm_analysis(cleaned_df)
        cust_df.to_sql('customer_datatable',con=engine,index=False,if_exists='replace')

        # MERCHANTS_DATATABLE
        print("extracting and cleaning merchants data table")
        cleaned_df = pd.read_sql_query('select * from clean_raw_data',con=engine)
        merchant_df = merchant_db.merchant_cleaning(cleaned_df)
        merchant_df.to_sql('merchant_frequency',con=engine,index=False,if_exists='replace')
        print("successfully uploaded merchants_frequency")
        
        return redirect('http://localhost:3000/flask/templates/cleanRawDataTable.html',code = 201)
        # return ('success',201)

    # return "error"

@app.route('/getdata', methods=['GET','POST']) # binds URL to view function
def getdata():
    if 'tablename' in request.args:
        try:
            tablename = str(request.args.get('tablename'))
            psql = "SELECT * FROM {};".format(tablename)
            df = pd.read_sql_query(psql,engine)
            result = df.to_json(orient="records")
            parsed = json.loads(result)
            return jsonify(parsed)
        except: 
            return ("Enter valid table name")
    return "tablename not found"

#this function runs the merchant twt scrapping and sentiment analysis from the merchant_frequency table
@app.route('/merchant', methods=['GET','POST']) # binds URL to view function
def merchant():
    df = pd.read_sql_query('SELECT * FROM merchant_frequency;',engine)
    final_merchant_df = merchant_db.sentiment_analysis(df)
    final_merchant_df.to_sql('merchant_score',con=engine,index=False,if_exists='replace')
    return ('success',201)

@app.route('/mba', methods=['POST']) # binds URL to view function
def mba():
    survey = pd.read_sql_query('SELECT * FROM clean_raw_data;',engine)
    df_categories = pd.read_sql_query('SELECT * FROM merchant_score;',engine)
    mba_df = mba_db.concat_categories(survey,df_categories)
    mba_df.to_sql('mba',con=engine,index=False,if_exists='replace')
    return ('success',201)

#this function is for the form data (Machine learning algo)
@app.route('/formdata', methods=['GET','POST']) # binds URL to view function
def formdata():
    age = str(request.args.get('age'))
    gender = str(request.args.get('gender'))
    region = str(request.args.get('region'))
    industry = str(request.args.get('industry'))
    annual_income = str(request.args.get('income'))
    education = str(request.args.get('education'))


    df = pd.read_sql_query('SELECT * FROM clean_raw_data;',engine)
    mba_db = pd.read_sql_query('SELECT * FROM mba;',engine)
    score_db = pd.read_sql_query('SELECT * FROM merchant_score;',engine)
    cust_db = pd.read_sql_query('SELECT * FROM customer_datatable;',engine)
    
    form_dict = {'gender':gender,
             'age': age,
             'annual_income': annual_income,
             'education': education,
             'region': region,
             'industry': industry}

    form = ML_form.convert_dict(form_dict)
    prediction_list = ML_form.cat_prediction(df, form)
    segment = ML_form.rfm_classification(cust_db, form)
    cat_merchant_dict = {}

    for cat in prediction_list:
        cat_merchant_dict[cat]= ML_form.mba_recommendation(mba_db, score_db, cat)

    print(cat_merchant_dict)

    json_response = [
                        {'predicted_rfm_segment':segment,
                        'category_merchant_pair':
                        cat_merchant_dict}
    ]
    
    json_response2 = jsonify(json_response)
    json_response2.headers.add('Access-Control-Allow-Origin', '*')
    return json_response2



if __name__ == '__main__': # if file is run as main program
    app.run(debug=True)
    # app.run(host="localhost", port=3000, debug=True)