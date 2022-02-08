

import pandas as pd
# df_reader = pd.read_json("Clothing_Shoes_and_Jewelry.json",lines=True,chunksize=1000000)
# counter = 1

# for chunk in df_reader:
#     new_df = pd.DataFrame(chunk[['overall','reviewText','summary']])
#     new_df1 = new_df[new_df['overall']==5].sample(8000)
#     new_df2 = new_df[new_df['overall']==4].sample(8000)
#     new_df4 = new_df[new_df['overall']==2].sample(8000)
#     new_df5 = new_df[new_df['overall']==1].sample(8000)
    
#     new_df6 = pd.concat([new_df1,new_df2,new_df4,new_df5], axis=0, ignore_index=True)
    
#     new_df6.to_csv(str(counter)+".csv",index=False)
#     new_df=None
#     counter = counter+1
    
# import pandas as pd
import numpy as np
# from glob import glob

# filename = glob("*.csv")
# dataframes = [pd.read_csv(f) for f in filename]

# frame = pd.concat(dataframes,axis=0,ignore_index=True)
# frame.to_csv('balanced_reviews.csv',index=False)


# data = pd.read_csv(r"balanced_reviews.csv")
# data.isnull().sum() 
# data[data.isnull().any(axis=1)] 
# data.dropna(inplace=True) 
# data = data[data['overall']!=3] 
# data['Positivity'] = np.where(data['overall'] > 3, 1, 0) 

# data.to_csv('final_reviews.csv',index=False)
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

import re
df= pd.read_csv("final_reviews.csv")
corpus = []
for i in range(0,1054658):
    review = re.sub('[^a-zA-Z]',' ', df.iloc[i,1])
    
    review = review.lower()
    
    review = review.split()
    
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review ]
    
    review = " ".join(review)
    
    corpus.append(review)
    print(i)
labels = df.iloc[0:1054658, 3] 

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer


features = CountVectorizer().fit_transform(corpus)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(features_train, labels_train)
#score
from sklearn.metrics import roc_auc_score

predictions = model1.predict(features_test)

print(roc_auc_score(labels_test, predictions))




# #TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer


vect = TfidfVectorizer().fit_transform(corpus)



features_train, features_test, labels_train, labels_test = train_test_split(vect, labels)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features_train, labels_train)
#score
from sklearn.metrics import roc_auc_score

predictions = model.predict(features_test)

print(roc_auc_score(labels_test, predictions))

import pickle

pkl_filename = "pickle_model1.pkl"


with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
    

