from utils import replace_newline, make_tokens, remove_stopwords, letters_only
from collections import Counter 
import spacy
import pickle
import pandas as pd 
from spacy.tokenizer import Tokenizer
import sys

# We want to extend our current list of Rule Matchers for Countries and Tax categories to include Rule Matchers for subcategories 
nlp = spacy.load('en_core_web_lg')
main_cat_matcher = spacy.matcher.Matcher(nlp.vocab)
main_cat_matcher.add('topic', None, [{'LOWER':'beps'}], 
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
# matcher.add('country', None, [{'ENT_TYPE':'GPE'}])[]

# additional subcategories 
direct_tax_matcher = spacy.matcher.Matcher(nlp.vocab)
direct_tax_matcher.add('direct_tax', None, [{'LOWER':'indirect'}, {'LOWER':'transfers'}],
            [{'LOWER':'US'}, {'LOWER':'trade'}],
            [{'LOWER':'US'}, {'LOWER':'business'}],
            [{'LOWER':'hybrid'}, {'LOWER':'instruments'}],
            [{'LOWER':'limitations'}, {'LOWER':'of'}, {'LOWER':'benefits'}],
            [{'LOWER':'effectively'}, {'LOWER':'connected'}, {'LOWER':'income'}],
            [{'LOWER':'loan'}, {'LOWER':'origination'}],
            [{'LOWER':'firpta'}],
            [{'LOWER':'tax'}, {'LOWER':'compliance'}],
            [{'LOWER':'us'}, {'LOWER':'tax'},{'LOWER':'reform'}],
            [{'LOWER':'trading'}, {'LOWER':'safe'}, {'LOWER':'harbour'}],
            [{'LOWER':'withholding'}, {'LOWER':'tax'}],
            [{'LOWER':'tax'}, {'LOWER':'planning'}],
            [{'LOWER':'tax'}, {'LOWER':'residence'}],
            [{'LOWER':'state'}, {'LOWER':'and'},{'LOWER':'local'},{'LOWER':'tax'}],
            [{'LOWER':'branch'}, {'LOWER':'profits'}, {'LOWER':'tax'}],
            [{'LOWER':'tax'}, {'LOWER':'budget'}],
            [{'LOWER':'tax'}, {'LOWER':'deductibility'}],
            [{'LOWER':'corporate'}, {'LOWER':'tax'}],
            [{'LOWER':'controlled'}, {'LOWER':'foreign'}, {'LOWER':'corporation'}],
            [{'LOWER':'net'}, {'LOWER':'operating'},{'LOWER':'losses'}],
            [{'LOWER':'tax'}, {'LOWER':'rates'}],
            [{'LOWER':'capital'}, {'LOWER':'gains'}, {'LOWER':'tax'}],
            [{'LOWER':'tax'}, {'LOWER':'incentive'}],
            [{'LOWER':'tax'}, {'LOWER':'reclaims'}])

transfer_pricing_matcher = spacy.matcher.Matcher(nlp.vocab)
transfer_pricing_matcher.add('transfer_pricing', None, 
            [{'LOWER':'dependent'}, {'LOWER':'agent'}, {'LOWER': 'pe'}], 
            [{'LOWER':'permanent'}, {'LOWER':'establishment'}],
            [{'LOWER':'debt'}, {'LOWER':'equity'}, {'LOWER':'ratio'}],
            [{'LOWER':'advanced'}, {'LOWER':'pricing'}, {'LOWER':'arrangement'}],
            [{'LOWER':'advanced'}, {'LOWER':'pricing'}, {'LOWER': 'agreement'}],
            [{'LOWER':'thin'}, {'LOWER':'capitalization'}],
            [{'LOWER':'interest'}, {'LOWER':'deductibility'}])

inst_invs_matcher = spacy.matcher.Matcher(nlp.vocab)
inst_invs_matcher.add('institutional_investors', None, [{'LOWER':'private'}, {'LOWER':'equity'}],
            [{'LOWER':'qualified'}, {'LOWER':'foreign'},{'LOWER':'pension'},{'LOWER':'fund'}],
            [{'LOWER':'commercial'}, {'LOWER':'activity'}],
            [{'LOWER':'commercial'}, {'LOWER':'activity'}, {'LOWER':'income'}],
            [{'LOWER':'sovereign'}, {'LOWER':'immunity'}],
            [{'LOWER':'funds'}],
            [{'LOWER':'real'}, {'LOWER':'estate'}],
            [{'LOWER':'sovereign'}, {'LOWER':'wealth'},{'LOWER':'fund'}],
            [{'LOWER':'enterprise'}, {'LOWER':'tax'}],
            [{'LOWER':'section'}, {'LOWER':'892'}],
            [{'LOWER':'sovereign'}, {'LOWER':'exemption'}],
            [{'LOWER':'secondaries'}],
            [{'LOWER':'listed'}, {'LOWER':'securities'}],
            [{'LOWER':'credit'}],
            [{'LOWER':'tax'}, {'LOWER':'due'}, {'LOWER':'dilligence'}],
            [{'LOWER':'infrastructure'}])

tax_func_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_func_matcher.add('tax_function_risk_management_and_strategy', None, [{'LOWER':'tax'}, {'LOWER':'function'}],
            [{'LOWER':'tax'}, {'LOWER':'governance'}, {'LOWER':'framework'}],
            [{'LOWER':'tax'}, {'LOWER':'strategy'}])

indirect_tax_matcher = spacy.matcher.Matcher(nlp.vocab)
indirect_tax_matcher.add('changes_indirect_tax', None, [{'LOWER':'output'}, {'LOWER':'tax'}],
            [{'LOWER':'transfer'}, {'LOWER':'tax'}],
            [{'LOWER':'goods'}, {'LOWER':'and'},{'LOWER':'services'}, {'LOWER':'tax'}],
            [{'LOWER':'gst'}],
            [{'LOWER':'property'}, {'LOWER':'tax'}],
            [{'LOWER':'securities'}, {'LOWER':'transaction'},{'LOWER':'tax'}],
            [{'LOWER':'value-added'}, {'LOWER':'tax'}],
            [{'LOWER':'input'}, {'LOWER':'tax'}],
            [{'LOWER':'stamp'}, {'LOWER':'duty'}],
            [{'LOWER':'landholder'}, {'LOWER':'duty'}],
            [{'LOWER':'tax'}, {'LOWER':'surcharge'}],
            [{'LOWER':'financial'}, {'LOWER':'transaction'},{'LOWER':'tax'}],
            [{'LOWER':'transaction'}, {'LOWER':'tax'}])

tax_transparency_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_transparency_matcher.add('tax_transparency', None, [{'LOWER':'mandatory'}, {'LOWER':'disclosure'}, {'LOWER':'rules'}],
            [{'LOWER':'country-by-country'}, {'LOWER':'reporting'}],
            [{'LOWER':'master'}, {'LOWER':'file'}],
            [{'LOWER':'local'}, {'LOWER':'file'}],
            [{'LOWER':'exchange'}, {'LOWER':'of'}, {'LOWER':'information'}],
            [{'LOWER':'foreign'}, {'LOWER':'account'}, {'LOWER':'tax'},{'LOWER':'compliance'}, {'LOWER':'act'}],
            [{'LOWER':'fatca'}],
            [{'LOWER':'common'}, {'LOWER':'reporting'}, {'LOWER':'standards'}],
            [{'LOWER':'crs'}],
            [{'LOWER':'eu'}, {'LOWER':'mandatory'}, {'LOWER':'disclosure'}, {'LOWER':'directive'}],
            [{'LOWER':'dac6'}],
            [{'LOWER':'exhange'}, {'LOWER':'of'}, {'LOWER':'tax'}, {'LOWER':'rulings'}],
            [{'LOWER':'anti-tax'}, {'LOWER':'avoidance'}, {'LOWER':'directive'}],
            [{'LOWER':'atad'}])

tax_tech_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_tech_matcher.add('tax_technology', None, [{'LOWER':'digital'}, {'LOWER':'tax'}],
            [{'LOWER':'robotic'}, {'LOWER':'process'}, {'LOWER':'automation'}],
            [{'LOWER':'blockchain'}],
            [{'LOWER':'tax'}, {'LOWER':'data'},{'LOWER':'management'}],
            [{'LOWER':'artificial'}, {'LOWER':'intelligence'}])

anti_avoidance_matcher = spacy.matcher.Matcher(nlp.vocab)
anti_avoidance_matcher.add('anti_avoidance', None, [{'LOWER':'anti-avoidance'}],
            [{'LOWER':'tax'}, {'LOWER':'avoidance'}],
            [{'LOWER':'general'}, {'LOWER':'anti-avoidance'},{'LOWER':'rule'}],
            [{'LOWER':'gaar'}],
            [{'LOWER':'anti-abuse'}, {'LOWER':'rules'}],
            [{'LOWER':'anti-hybrid'}, {'LOWER':'rules'}])

beps_matcher = spacy.matcher.Matcher(nlp.vocab)
beps_matcher.add('beps', None, [{'LOWER':'treaty'}, {'LOWER':'shopping'}],
            [{'LOWER':'tax'}, {'LOWER':'substance'}],
            [{'LOWER':'double'}, {'LOWER':'tax'}, {'LOWER':'treaty'}],
            [{'LOWER':'principal'}, {'LOWER':'purpose'}, {'LOWER':'test'}],
            [{'LOWER':'un'}, {'LOWER':'model'}, {'LOWER':'treaty'}],
            [{'LOWER':'beneficial'}, {'LOWER':'owner'}],
            [{'LOWER':'multilateral'}, {'LOWER':'instrument'}],
            [{'LOWER':'economic'}, {'LOWER':'substance'}],
            [{'LOWER':'tax'}, {'LOWER':'treaty'},{'LOWER':'ramification'}],
            [{'LOWER':'business'}, {'LOWER':'purpose'},{'LOWER':'test'}],
            [{'LOWER':'european'}, {'LOWER':'union'}])

tax_controversy_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_controversy_matcher.add('tax_controversy', None, [{'LOWER':'tax'}, {'LOWER':'dispute'}],
            [{'LOWER':'tax'}, {'LOWER':'rulings'}],
            [{'LOWER':'tax'}, {'LOWER':'audit'}],
            [{'LOWER':'hybrid'}, {'LOWER':'entity'}],
            [{'LOWER':'tax'}, {'LOWER':'litigation'}],
            [{'LOWER':'tax'}, {'LOWER':'scrutiny'}],
            [{'LOWER':'tax'}, {'LOWER':'evasion'}],
            [{'LOWER':'facilitation'}, {'LOWER':'of'},{'LOWER':'tax'}, {'LOWER':'evasion'}],
            [{'LOWER':'tax'}, {'LOWER':'refunds'}],
            [{'LOWER':'standard'}, {'LOWER':'audit'},{'LOWER':'file'}, {'LOWER':'for'}, {'LOWER':'tax'}],
            [{'LOWER':'saf-t'}],
            [{'LOWER':'mutual'}, {'LOWER':'agreement'}, {'LOWER':'procedure'}],
            [{'LOWER':'tax'}, {'LOWER':'authority'}],
            [{'LOWER':'tax'}, {'LOWER':'penalities'}],
            [{'LOWER':'statute'}, {'LOWER':'of'},{'LOWER':'limitations'}])
# apply matcher to htmls 

def pipeline(document):
    return replace_newline(document.lower())

def process_text(texts):
    clean_texts = [pipeline(doc) for doc in texts]
    tokenizer = Tokenizer(nlp.vocab)
    tokenized_texts = [tokenizer(text) for text in clean_texts]
    return tokenized_texts
   
 # create list of matches
def get_matches(matcher, tokenized_texts):  
    ls = []
    for text in matcher.pipe(tokenized_texts):
        matches = matcher(text)
        cats_ls = []
        for _, start, end in matches:
            span = text[start:end]
            span_text = span.text
            cats_ls.append(span_text)
        ls.append(cats_ls)
    return ls    

# read in data 
#with open('../data/scraped_htmls.pickle', 'rb') as f:
#    htmls = pickle.load(f)
#df = pd.DataFrame(htmls) 
df=pd.read_csv('/data/home/gicpoc/gic/data_sets/test1.csv')
df=pd.read_csv(sys.argv[1])
#df_valid=pd.read_csv('/data/home/gicpoc/gic/data_sets/gic_poc_validate.csv')

df['textstr']=df['text'].astype(str)
texts = df['textstr']
#df_valid['textstr']=df_valid['text'].astype(str)
#texts_valid=df_valid['textstr']
# clean and tokenize text
tokenized_texts = process_text(texts)
#tokenized_texts_valid=process_text(texts_valid)

# match subcategory to text 
matcher_list = [direct_tax_matcher, transfer_pricing_matcher, inst_invs_matcher, tax_func_matcher, indirect_tax_matcher, tax_transparency_matcher,
                tax_tech_matcher, anti_avoidance_matcher, beps_matcher, tax_controversy_matcher]

columns=['directTax','transferPricing','instInv', 'taxFunc', 'indirectTax', 'taxTrans', 'taxTech', 'antiAvoidance','beps', 'taxControv']
subcats = pd.DataFrame(columns=columns)
#subcats_valid=pd.DataFrame(columns=columns)


for matcher, category in zip(matcher_list, columns): 
    got_matches = get_matches(matcher, tokenized_texts)
    match_count = [Counter(doc) for doc in got_matches]
    print(len(match_count))
    subcats[category]=match_count
#    subcats=pd.concat([subcats, match_count], axis=1, ignore_index=True)
subcats['filename']=df['Reference']
#subcats['filename']=df['filename']
#subcats.to_pickle('/data/home/gicpoc/gic/tmp_poc/v1_matcher_test.pkl')
subcats.to_pickle(sys.argv[2])


#for matcher, category in zip(matcher_list, columns): 
#    got_matches = get_matches(matcher, tokenized_texts_valid)
#    match_count = [Counter(doc) for doc in got_matches]
#    print(len(match_count))
#    subcats_valid[category]=match_count
#subcats_valid['filename']=df_valid['filename']
#subcats_valid.to_pickle('/data/home/gicpoc/gic/tmp_poc/v1_matcher_validate.pkl')

def counter2ls(df_column):
    ls_col = [list(doc.items()) for doc in df_column]
    return ls_col
