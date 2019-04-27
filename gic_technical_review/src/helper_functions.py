import nltk
import pandas as pd
import spacy

def replace_newline(text):
    return text.replace('\n', ' ')

def make_tokens(text):
    return [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)] #I DON'T KNOW WHY I DO THIS. WHY NOT WORD_TOKENIZE DIRECTLY?

def remove_stopwords(tokens):
    stop = nltk.corpus.stopwords.words('english')
    stop.append('The')
    stop.append('A')
    stop.append('This')
    stop.append('KPMG')
    stop.append('PwC')
    stop.append('EY')
    stop.append('Deloitte')
    tokens = [x for x in tokens if x not in stop]
    return tokens

def lemming(tokens):
    wnl = nltk.WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]
    return tokens

def letters_only(tokens):
    tokens = [t for t in tokens if t.isalnum()]
    return tokens

def alphanum(tokens):
    tokens = [t for t in tokens if t.isalpha()]
    return tokens

def make_bigrams(tokens):
    return list(nltk.bigrams(tokens))

def basic_dataframe(pipeline, full_dataset):
	# Process data
	all_text=[]
	all_label=[]
	all_filename=[]
	for ii, subset in enumerate(full_dataset):
	    all_text.append(pipeline(subset['text']))
	    all_label.append(subset['category'])
	    all_filename.append(subset['filename'])
	# Convert to pandas dataframe
	df_docs = pd.DataFrame({'text':all_text, 'category':all_label, 'filename':all_filename})
	df_docs['category'] = pd.Categorical(df_docs['category'])
	df_docs['target'] = df_docs['category'].cat.codes
	print(dict(enumerate(df_docs['category'].cat.categories)))
	def func(the_list):
	    return ' '.join(the_list)
	df_docs['text_string'] = df_docs.apply(lambda x: func(x['text']),axis=1)
	return df_docs

def matcher_v1(nlp):
	matcher = spacy.matcher.Matcher(nlp.vocab)
	matcher.add('topic', None, [{'LOWER':'beps'}], 
	            [{'LOWER':'transfer'}, {'LOWER':'pricing'}], 
	           [{'LOWER':'tax'},{'LOWER':'technology'}], 
	           [{'LOWER':'institutional'}, {'LOWER':'investors'}], 
	           [{'LOWER':'risk'}, {'LOWER':'management'}], 
	           [{'LOWER':'indirect'}, {'LEMMA':'tax'}], 
	           [{'LOWER':'tax'}, {'LOWER':'controversy'}], 
	           [{'LOWER':'direct'}, {'LOWER':'tax'}], 
	           [{'LOWER':'anti'}, {}, {'LOWER':'avoidance'}], 
	            [{'LOWER':'tax'}, {'LOWER':'avoidance'}], 
	           [{'LOWER':'tax'}, {'LOWER':'transparency'}])
	matcher.add('country', None, [{'ENT_TYPE':'GPE'}])
	return matcher

def countriesDict_v1():
	countriesDict = {}
	for key in ['uk', 'united kingdom']:
	    countriesDict[key] = 'UK'
	for key in ['us', 'usa', 'united states', 'united states of america', 'the united states', 'u.s', 'u.s.a', 'u.s.']:
	    countriesDict[key] = 'USA'
	for key in ['brazil']:
	    countriesDict[key] = 'Brazil'
	for key in ['france']:
	    countriesDict[key] = 'France'
	for key in ['spain']:
	    countriesDict[key] = 'Spain'
	for key in ['china']:
	    countriesDict[key] = 'China'
	for key in ['korea', 'south korea']:
	    countriesDict[key] = 'Korea'
	for key in ['japan']:
	    countriesDict[key] = 'Japan'
	for key in ['australia']:
	    countriesDict[key] = 'Australia'
	for key in ['india']:
	    countriesDict[key] = 'India'
	return countriesDict

cat_matcher= {'anti-avoidance':0, 'tax avoidance':0, 'anti tax avoidance':0, 'beps':1, 'direct tax':2, 'indirect tax':3, \
             'indirect taxes':3, 'institutional investors':4, \
            'risk management':5, 'tax controversy':6, 'tax technology':7, 'tax transparency':8, 'transfer pricing':9}

gpe_matcher= {'UK':0, 'USA':1, 'Brazil':2, 'France':3, 'Spain':4, 'China':5, \
            'Korea':6, 'Japan':7, 'Australia':8, 'India':9}

dictofcat={0: 'anti-avoidance',
 1: 'base_erosion_profit_sharing',
 2: 'direct_tax',
 3: 'indirect_tax',
 4: 'institutional_investors',
 5: 'risk_management_strat',
 6: 'tax_controversy',
 7: 'tax_technology',
 8: 'tax_transparency',
 9: 'transfer_pricing'}

dictofgpe={0: 'australia',
 1: 'brazil',
 2: 'china',
 3: 'france',
 4: 'india',
 5: 'japan',
 6: 'korea',
 7: 'spain',
 8: 'uk',
 9: 'usa'}