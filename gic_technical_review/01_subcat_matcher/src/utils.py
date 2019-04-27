
import nltk
import pandas as pd
import spacy

def replace_newline(text):
    return text.replace('\n', ' ')

def make_tokens(text):
    return [sent for sent in nltk.sent_tokenize(text)] #I DON'T KNOW WHY I DO THIS. WHY NOT WORD_TOKENIZE DIRECTLY?

def remove_stopwords(tokens):
    stop = nltk.corpus.stopwords.words('english')
    stop.append('The')
    stop.append('A')
    stop.append('This')
    tokens = [x for x in tokens if x not in stop]
    return tokens

def lemming(tokens):
    wnl = nltk.WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]
    return tokens

def letters_only(tokens):
    tokens = [t for t in tokens if t.isalpha()]
    return tokens

def make_bigrams(tokens):
    return list(nltk.bigrams(tokens))


