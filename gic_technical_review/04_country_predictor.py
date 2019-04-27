import numpy as np
import spacy
from spacy.pipeline import EntityRuler
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd 

# write rule matcher patterns to augment existing NER classifier with special cases of country mentions eg. hmrc refers to the uk 

patterns = [
        {"label":"GPE", "pattern":[{"lower":"american"}]},  # code proper adjectives as countries
        {"label":"GPE", "pattern":[{"lower":"america"}]},
        {"label":"GPE", "pattern":[{"lower":"us"}]},
        {"label":"GPE", "pattern":[{"lower":"u.s."}]},
        {"label":"GPE", "pattern":[{"lower":"united states"}]},
        {"label":"GPE", "pattern":[{"lower":"usa"}]},
        {"label":"GPE", "pattern":[{"lower":"u.s.a"}]},
        {"label":"GPE", "pattern":[{"lower":"brazilian"}]},
        {"label":"GPE", "pattern":[{"lower":"brazil"}]},
        {"label":"GPE", "pattern":[{"lower":"france"}]},
        {"label":"GPE", "pattern":[{"lower":"french"}]},
        {"label":"GPE", "pattern":[{"lower":"spain"}]},
        {"label":"GPE", "pattern":[{"lower":"spanish"}]},
        {"label":"GPE", "pattern":[{"lower":"china"}]},
        {"label":"GPE", "pattern":[{"lower":"chinese"}]},
        {"label":"GPE", "pattern":[{"lower":"korea"}]},
        {"label":"GPE", "pattern":[{"lower":"south korea"}]},
        {"label":"GPE", "pattern":[{"lower":"korean"}]},
        {"label":"GPE", "pattern":[{"lower":"japanese"}]},
        {"label":"GPE", "pattern":[{"lower":"japan"}]},
        {"label":"GPE", "pattern":[{"lower":"australia"}]}, 
        {"label":"GPE", "pattern":[{"lower":"australian"}]},
        {"label":"GPE", "pattern":[{"lower":"indian"}]},
        {"label":"GPE", "pattern":[{"lower":"britain"}]},
        {"label":"GPE", "pattern":[{"lower":"british"}]},
        {"label":"GPE", "pattern":[{"lower":"u.k."}]},
        {"label":"GPE", "pattern":[{"lower":"uk"}]},
        {"label":"GPE", "pattern":[{"lower":"united kingdom"}]},
        {"label":"GPE", "pattern":[{"lower":"hmrc"}]},  # code tax departments from main countries",
        {"label":"GPE", "pattern":[{"lower":"chancellor"}]},
        {"label":"GPE", "pattern":[{"lower":"treasury department"}]},
        {"label":"GPE", "pattern":[{"lower":"irs"}]},
        {"label":"GPE", "pattern":[{"lower":"the department of federal revenue of brazil"}]},
        {"label":"GPE", "pattern":[{"lower":"state administration of taxation"}]}]

# text preprocessing functions
def replace_newline(text):
    return text.replace('\n', ' ')

# extract geo-political entities from text and filter to include only countries of interest
def country_as_gpe(doc):
    country_ls = ['uk','u.k.','united kingdom','british','hmrc','chancellor',
          'usa','us','u.s','american','treasury department','irs',
          'Brazil','brazil','brazilian','the department of federal revenue of brazil',
          'france','french',
          'spain','spanish',
          'china','chinese','state administration of taxation',
          'korea','korean',
          'japan', 'japanese',
          'australia','australian',
          'india', 'indian']
    country_dict = {}
    for key in ['uk','u.k.','united kingdom','british','hmrc','chancellor']:
        country_dict[key] = 'UK'
    for key in ['usa','us','u.s','american','treasury department','irs']:
        country_dict[key] = 'US'
    for key in ['Brazil','brazil','brazilian','the department of federal revenue of brazil']:
        country_dict[key] = 'Brazil'
    for key in ['france','french']:
        country_dict[key] = 'France'
    for key in ['spain','spanish']:
        country_dict[key] = 'Spain'
    for key in ['china','chinese','state administration of taxation']:
        country_dict[key] = 'China'
    for key in ['korea','korean']:
        country_dict[key] = 'Korea'
    for key in ['japan', 'japanese']:
        country_dict[key] = 'Japan'
    for key in ['australia','australian']:
        country_dict[key] = 'Australia'
    for key in ['india', 'indian']:
        country_dict[key] = 'India'
    country = [ent.text for ent in doc.ents if ent.label_=='GPE']  # extract geo-political entities from document. 
    country = [c for c in country if c in country_ls]
    country = [country_dict[c] for c in country]
    return country

def df_predictions(docs, clean_texts):
    gpe_countries = [list(set(country_as_gpe(doc))) for doc in docs]
    df = pd.DataFrame({'gpe_countries':gpe_countries,'text':clean_texts})
    return df

def df2multilabels(df):
    multilabel_map = {'Australia':0,'Brazil':1,'China':2,'France':3,'India':4,'Japan':5,'Korea':6,'Spain':7,'UK':8,'US':9}
    multilabel = []
    for country in df['gpe_countries']:
        multilabel_ls = [multilabel_map[x] for x in country]
        multilabel.append(multilabel_ls)
    return multilabel

def multilabel_binarizer(multilabel):
    mlb = MultiLabelBinarizer()
    multilabel = mlb.fit_transform(multilabel)
    return multilabel

def main():
    # load_data
    test1_filepath = 'data/test1.csv'
    test2_filepath = 'data/test2.csv'
    test3_filepath = 'data/test3.csv'
    
    test1 = pd.read_csv(test1_filepath, index_col=0)
    test2 = pd.read_csv(test2_filepath, index_col=0)
    test3 = pd.read_csv(test3_filepath, index_col=0)
    print('loaded data')
    
    # remove unneccesary characters 	
    clean_texts_1 = [replace_newline(t.lower()) for t in test1['text']] 
    clean_texts_2 = [replace_newline(t.lower()) for t in test2['text']]
    clean_texts_3 = [replace_newline(t.lower()) for t in test3['text']]
    
    # apply tokenizer, tagger, parser and ner to text
    nlp = spacy.load('en_core_web_lg')
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)
    print('entity ruler added')

    test1_docs = [doc for doc in nlp.pipe(clean_texts_1, batch_size = 10)]
    test2_docs = [doc for doc in nlp.pipe(clean_texts_2, batch_size = 10)]
    test3_docs = [doc for doc in nlp.pipe(clean_texts_3, batch_size = 10)]    
    
    test1 = multilabel_binarizer(df2multilabels(df_predictions(test1_docs, clean_texts_1)))
    # test1 = np.insert(test1, 6, 0, axis=1)
    test1 = pd.DataFrame(test1, columns=['Australia','Brazil','China','France','India','Japan','Korea','Spain','UK','US']) 
    test1.to_csv('data/test1_country_predictions.csv')
    print('test1 predictions saved')
    
    test2 = multilabel_binarizer(df2multilabels(df_predictions(test2_docs, clean_texts_2)))
    test2 = pd.DataFrame(test2, columns=['Australia','Brazil','China','France','India','Japan','Korea','Spain','UK','US']) 
    test2.to_csv('data/test2_country_predictions.csv')
    print('test2 predictions saved')

    test3 = multilabel_binarizer(df2multilabels(df_predictions(test3_docs, clean_texts_3)))
    test3 = np.insert(test3, 5, 0, axis=1)
    test3 = pd.DataFrame(test3, columns=['Australia','Brazil','China','France','India','Japan','Korea','Spain','UK','US'])
    test3.to_csv('data/test3_country_predictions.csv')
    print('test3 predictions saved')

if __name__=="__main__":
    main()
