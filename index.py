# For Running ML Model
from matplotlib import lines
from prometheus_client import Summary
from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import  sklearn
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA

# For Parsing Page Content and Website
from flask import Flask, redirect, url_for, render_template, request
from bs4 import BeautifulSoup
from unicodedata import category
import http
import smtplib
import requests
import json

# from requests_html import HTMLSession
# from autoscraper import AutoScraper

from nltk.corpus import stopwords 
ENGLISH_STOP_WORDS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords') 
from sklearn.preprocessing import MinMaxScaler

 
# # Loading My Trained ML Model  
filename = 'final_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))

model = joblib.load('kmeans_model25.pkl')
 


# Website Code:-

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/output", methods=["POST", "GET"])
def output():
    if request.method == "POST":
       return redirect("/")  
    else:
       return render_template("index.html")

#  My Main Code
@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        url = request.form["urlname"] # Taking Page URL
         
 
        
        # Code For Parsing HTML page and finding the INPUT data for Model:-

        headers = { "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/74.0.3729.169 Safari/537.36"}

        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        File = open("out.csv", "a") 
        divTag = soup.find_all("div",class_="a-section review aok-relative")
        count = len(divTag)
        outfile= open("sample.json","w") 
  
        myMap = {"January":1,"February":2,"March":3,"April":4,"May":5,
                 "June":6,"July":7,"August":8,"September":9,"October":10,
                 "November":11,"December":12} 
                
        list_1 = list()

 
    
       
 #    As My Model is trained for 9 attributes :- 
 #    So I am scrabing one by one from HTML page using For Loop
    
        outputdic=[]

        for i in range(count):
            div=divTag[i]['id']
            print('================ div id:-> '+div)
            div=soup.find(id=div)
# 1 verall-------------------
            item =div.find("span", class_="a-icon-alt")
            overall_text=item.string.strip().replace(',', '')
            
            overall=overall_text[0]
            File.write(f"{overall},")
            overall=int(overall)
            list_1.append(overall)
            print(overall)
            
          
# 2 verified --------------
            item =div.find("span", class_="a-size-mini a-color-state a-text-bold")
            verified_text=item.string.strip().replace(',', '')
            # File.write()
            if verified_text == "Verified Purchase":
               verified=True
               Verified=1
            else:
               Verified=0
               verified=False
            File.write(f"{verified},")
            # verified=int(verified)
            print(verified)
             

 # 3 date ---------------
            item =div.find("span",class_="a-size-base a-color-secondary review-date")   
            date_text=item.string.strip().replace(',', '') 
            print ("-------")
            print (date_text)
            date_text = date_text.split('on ', 1)
            date_text=date_text[1]
            print (date_text)
            datecount=len(date_text)
            
            tempvar=""
            for j in range(datecount):
                if date_text[j]==' ':
                   break
                else:
                   tempvar=tempvar+date_text[j]

            date=tempvar
            date=int(date)
            
            tempvar=""
            count2=0
            for j in range(datecount):
                if count2==2:
                    break
                elif date_text[j]==' ':
                    count2=count2+1
                elif count2==1:
                    tempvar=tempvar+date_text[j]

            month= myMap.get(tempvar)
            
            tempvar=""
            count2=0
            for j in range(datecount):
                if count2>=2:
                    tempvar=tempvar+date_text[j]
                elif date_text[j]==' ':
                    count2=count2+1

            year=tempvar   
            
            
            month=int(month)
            File.write(f"{month} ")
            File.write(f"{date} ")
            File.write(f"{year},")

            print(date)
            print(month)
            print(year)

            # res = date_text.split('on ', 1)
            # res=str(res)
            # temp_date = res[1]
            # date = temp_date.partition(' ')[0]
            # temp_date = temp_date.split(' ', 1)
            # temp_date=str(temp_date)
            # temp_month = temp_date.partition(' ')[0]
            # # month=myMap[temp_month]
            # month= myMap.get(temp_month) 
            # print(date)
            # print(month)  
            # date=date_text[dlocation+3:dlocation+14])
 
# 4 c_ID -------------------
            customerTag = div.find("a",class_="a-profile")
            clink=customerTag['href']   # --> finding id using link
            clocation=clink.find('account.') 
            # print(clink[clocation+8:clocation+36])
            customerid=clink[clocation+8:clocation+36]
            print(customerid)
            File.write(f"{customerid},")
            # File.write(f"{clink[clocation+8:clocation+36]},")

 # 5 p_ID ---------------
         # for product id from link
            productTag = soup.find_all("a",class_="a-link-normal")
            plink=productTag[0]['href'] 
            # print(plink) 
            plocation=plink.find('/dp/')
            # print(plink[plocation+4:plocation+14])
            # print(plink[plocation+4:plocation+14])
            productid=plink[plocation+4:plocation+14]
            print(productid)
            File.write(f"{productid},")
            # File.write(f"{plink[plocation+4:plocation+14]}\n")
# 6 c_name ----------------
            item=div.find("span",class_="a-profile-name")
            # print(item.get_text())
            customername=item.string.strip().replace(',', '')
            print(customername)
            File.write(f"{customername},")
            # File.write(f"{item.string.strip().replace(',', '')},")
# 7 title -----------------
            item = div.find("span", class_="")
            # print(item.get_text()) 
            review=item.string.strip().replace(',', '')
            File.write(f"{review},")
            review_word_count=len(review)
            print(review)
            review_word_count=int(review_word_count)
            print(review_word_count)

            # File.write(f"{item.string.strip().replace(',', '')},")
            
# 7 summary -----------------
            item = div.find("span", class_="")
            # print(item.get_text()) 
            summary=item.string.strip().replace(',', '')
            File.write(f"{summary},")
            summary_word_count=len(summary)
            print(summary_word_count)
            print(summary)
            summary_word_count=int(summary_word_count)
            # File.write(f"{item.string.strip().replace(',', '')},")
            
# 9 voting --------------
            item =div.find("span",class_="a-size-base a-color-tertiary cr-vote-text")
            voting_text=item.string.strip().replace(',', '')
            # File.write(f"{item.string.strip().replace(',', '')},")
            vote = voting_text.split(" people")[0]
            File.write(f"{vote}\n")
            vote=int(vote)
            print(vote)

            data=[[overall,Verified,vote,review_word_count,summary_word_count,month,date]]
            clust=loaded_model.predict(data)[0] 

            fakecheck=0
            if clust == 25 or clust == 18:
                fakecheck=1
                v = {customername:fakecheck}
                outputdic.append(v)
            else:
                fakechceck=0
                v = {customername:fakecheck}
                outputdic.append(v)

                # ------------------------------------------------------------------------------
 
        File.close()

        dataframe=pd.read_csv('out.csv')

        def cleanDF(dataframe):
                # print(modified_df.head(1))
                modified_df = dataframe.dropna(axis = 0, subset = ['reviewText'])  
                # modified_df['vote'] = modified_df['vote'].str.replace(',','') 
                modified_df['vote'] = modified_df['vote'].fillna(0).astype(int) 
                modified_df['summary'].fillna(modified_df['reviewText'], inplace = True) 
                modified_df['reviewerName'].fillna('Amazon Customer', inplace = True) 
                modified_df = modified_df.astype({'reviewTime': 'datetime64[ns]'}) 
                modified_df['verified'] = modified_df['verified'].astype(int) 
                modified_df.drop_duplicates(inplace=True) 
                modified_df = modified_df.reset_index().drop('index', axis=1) 
                return modified_df



        def featureEngin(dataframe): 
              dataframe['review_word_count'] = dataframe['reviewText'].str.split().str.len()
              dataframe['summary_word_count'] = dataframe['summary'].str.split().str.len() 
              dataframe['month'] = pd.DatetimeIndex(dataframe['reviewTime']).month 
              dataframe['dayofweek'] = pd.DatetimeIndex(dataframe['reviewTime']).dayofweek
              map_numreviews = dataframe['reviewerID'].value_counts().to_dict() 
              dataframe['multipleReviews_reviewer'] = dataframe['reviewerID'].map(map_numreviews) 
              dataframe['multipleReviews_reviewer'] = np.where(dataframe['multipleReviews_reviewer'] > 1, 1, 0)
    
              map_five = dataframe['overall'].groupby(dataframe['reviewerID']).agg(lambda x: (np.unique(x)==5).all()).to_dict()
              map_one = dataframe['overall'].groupby(dataframe['reviewerID']).agg(lambda x: (np.unique(x)==1).all()).to_dict() 
              dataframe['reviewer_five_star_only'] = dataframe['reviewerID'].map(map_five)
              dataframe['reviewer_one_star_only'] = dataframe['reviewerID'].map(map_one)
              dataframe['reviewer_five_star_only'] = dataframe['reviewer_five_star_only'].astype(int)
              dataframe['reviewer_one_star_only'] = dataframe['reviewer_one_star_only'].astype(int)
              map_numreviews = dataframe['asin'].value_counts().to_dict()
              dataframe['numReviews_product'] = dataframe['asin'].map(map_numreviews) 
              dataframe['nameProvided'] = np.where(dataframe['reviewerName'] != 'Amazon Customer',1,0)
              return dataframe



        # def pipedf(path):
        dataframe=cleanDF(dataframe)
        # return featureEngin(cleanDF(createPdDF(path))) 
        df=featureEngin(dataframe)


        def spl_tokenizer(sentence): 
                for punctuation_mark in string.punctuation: 
                    sentence = sentence.replace(punctuation_mark,'').lower() 
                listofwords = sentence.split(' ')
                listoflemmatized_words = [] 
                for word in listofwords:
                   if (not word in ENGLISH_STOP_WORDS) and (word!=''): 
                      token = WordNetLemmatizer().lemmatize(word) 
                      try:
                         if tfidf.type == 'review':
                              token = 'r_' + token
                         elif tfidf.type == 'summary':
                              token = 's_' + token
                      except:
                           pass 
                      listoflemmatized_words.append(token) 
                return listoflemmatized_words



        def sps_tokenizer(sentence):
    
                for punctuation_mark in string.punctuation: 
                   sentence = sentence.replace(punctuation_mark,'').lower()

     
                listofwords = sentence.split(' ')
                listofstemmed_words = [] 
                stemmer = PorterStemmer()  
                for word in listofwords:
                    if (not word in ENGLISH_STOP_WORDS) and (word!=''): 
                            token = stemmer.stem(word)  
                            try:
                                if tfidf.type == 'review':
                                 token = 'r_' + token
                                elif tfidf.type == 'summary':
                                  token = 's_' + token
                            except:
                                  continue 
                            listofstemmed_words.append(token)

                return listofstemmed_words




        def tfidf(dataframe_column, tokenizer, min_df=0.02, max_df=1, ngram_range=(1,1)): 
                column_name = dataframe_column.name 
                if column_name == 'reviewText':
                    tfidf.type = 'review'
                elif column_name == 'summary':
                    tfidf.type = 'summary'
                else:
                    tfidf.type = 'none' 

                vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df, tokenizer = tokenizer, ngram_range = ngram_range) 
                vectorizer.fit(dataframe_column) 
                reviews_tokenized = vectorizer.transform(dataframe_column) 
                tokens = pd.DataFrame(columns=vectorizer.get_feature_names(), data=reviews_tokenized.toarray())
    
                return tokens


        
        
        # df = pipedf('out.csv')

        review_tokens = tfidf(df['reviewText'], tokenizer=spl_tokenizer, ngram_range=(1,2))
        summary_tokens = tfidf(df['summary'], tokenizer=spl_tokenizer, ngram_range=(1,2))

        df2=df.select_dtypes(include=['int32','int64'])


        mm = MinMaxScaler()
        df3 = mm.fit_transform(df2)
        df_scaled = pd.DataFrame(df3, columns = df2.columns)
        del df3, df2

        df_final = pd.concat([df_scaled, review_tokens, summary_tokens], axis = 1)
        del review_tokens, summary_tokens, df_scaled, df


        df_final.to_csv("out1.csv", encoding='utf-8')

        
        # pca = PCA(n_components=350)
        # pcs = pca.fit_transform(df_final)
        # PCA_components = pd.DataFrame(pcs)
        # finalcluster=model.predict(PCA_components.iloc[:,0:350])
        # print("===================================")
        # print(finalcluster)
        # print("===================================")
    
    # -----------------------------------------------------------------------
 
        
        outfile.close()
        error=""
        size=len(outputdic)
        #  {'error': error_message}
        # return render_template("index.html")  # return redirect(url_for("user", usr=user)) 
 
        return render_template("output.html",datas=outputdic,length=size,list=list_1)       
    else:
       return redirect("/")
         
        
 

if __name__ == "__main__":
    app.run(debug=True)

 
 



    