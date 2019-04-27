'''
This script generates probabilities for the 27 sub-categories.
Before running this script, run the spacy matcher to generate category matches that will be ingested by this script.
-Lik
'''

from nltk.cluster import KMeansClusterer
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import normalize
from scipy import sparse
from src.helper_functions import replace_newline, make_tokens, remove_stopwords, alphanum

test_data_path='./data/test3.csv'
save_probabilities_path='./data/test3_probabilities.csv'

vectorizer_path='./data/v1_vectorizer.npz'
tfidfier_path='./data/v1_tfidfier.npz'
matcher10cat_path='./data/v1_matcher_test.pkl'
matcher27cat_path='./data/v1_matcher_27_test.pkl'
pretrained_model_path='./data/v1_fine_tuned_models_v2.pkl'
clusterer_path='./data/v1_fine_tuned_clusterer.pkl'

#define columns
columns=['(Management of a) Tax function', 'Anti-Tax Avoidance Directive (ATAD)',
       'Controlled Foreign Corporation', 'Corporate Tax',
       'Country-by-Country Reporting', 'Digital Tax', 'Double Tax Treaty',
       'EU Mandatory Disclosure Directive (DAC6)', 'Economic substance',
       'General anti-avoidance rule (GAAR)', 'Goods and services tax (GST)',
       'Interest deductibility', 'Local file', 'Mandatory disclosure rules',
       'Master file', 'Multilateral Instrument (MLI)',
       'Mutual agreement procedure', 'Permanent Establishment (PE)',
       'Principal purpose test', 'Section 892', 'Sovereign Immunity',
       'Tax Governance Framework', 'Tax audit', 'Tax compliance',
       'Tax dispute', 'Value-added tax (VAT)', 'Withholding Tax']

# Load test text and trained vectorizer
vectorizer=pickle.load(open(vectorizer_path, 'rb'))
tfidfier=pickle.load(open(tfidfier_path, 'rb'))
df_test=pd.read_csv(test_data_path)

def preprocess(df):
    df['textstr']=df['text'].astype(str)
    df['textstr']=df.apply(lambda x: ' '.join(alphanum(remove_stopwords(make_tokens(replace_newline(x['textstr']))))), axis=1)
    return df

def make_features(X_valid, clusterer):
    n_clusters=clusterer._num_means
    X_valid_cat=np.array([clusterer.classify(row) for row in X_valid])
    X_valid_cat=np.eye(n_clusters)[X_valid_cat]
    X_valid_new=np.append(X_valid, X_valid_cat, axis=1)
    return X_valid_new

def generate_probabilities(all_model_outs, X_valid_new):
    probabilities=[]
    for model in all_model_outs:
        probabilities.append(model.predict_proba(X_valid_new))
    probabilities=np.mean(probabilities, axis=0)  
    return probabilities

# Vectorize text and load additional features
print('Processing text...')
df_test=preprocess(df_test)
data_features_test=vectorizer.transform(df_test['textstr'])
tfidf_test=tfidfier.transform(data_features_test)
df_counter_test=pd.read_pickle(matcher10cat_path)
df_counter_test_27=pd.read_pickle(matcher27cat_path)

# Generate features
def funct(x):
    return sum(x.values())
counter_test=normalize(df_counter_test.drop(labels='filename', axis=1).applymap(funct), axis=1, norm='l1')
counter_test27=normalize(df_counter_test_27.drop(labels='filename', axis=1).applymap(funct), axis=1, norm='l1')
X_test=np.concatenate((tfidf_test.toarray(), np.concatenate((counter_test, counter_test27), axis=1)), axis=1)

# Load trained models and clusterer
print('Loading models and predicting...')
all_model_outs2=pickle.load(open(pretrained_model_path, 'rb'))
clusterer=pickle.load(open(clusterer_path, 'rb'))

# Predict probabilities and save output
df_probabilities_test=pd.DataFrame(generate_probabilities(all_model_outs2, make_features(X_test, clusterer)), columns=columns)
df_probabilities_test.to_csv(save_probabilities_path)
print('Finished! ouput saved at ', save_probabilities_path)
