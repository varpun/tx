from utils import replace_newline, make_tokens, remove_stopwords, letters_only
from collections import Counter 
import spacy
import pickle
import pandas as pd 
from spacy.tokenizer import Tokenizer
import sys

# We want to extend our current list of Rule Matchers for Countries and Tax categories to include Rule Matchers for subcategories 
nlp = spacy.load('en_core_web_lg')

management_tax_function_matcher = spacy.matcher.Matcher(nlp.vocab)
management_tax_function_matcher.add('(Management of a) Tax function', None, 
            [{'LOWER':'tax'}, {'LOWER':'function'}],
            [{'LOWER':'tax'}, {'LOWER':'administrator'}],
            [{'LOWER':'tax'}, {'LOWER':'administrators'}],
            [{'LOWER':'tax'}, {'LOWER':'management'}],
            [{'LOWER':'tax'}, {'LOWER':'functionality'}])

atad_matcher = spacy.matcher.Matcher(nlp.vocab)
atad_matcher.add('Anti-Tax Avoidance Directive (ATAD)', None, 
            [{'LOWER':'anti-tax'}, {'LOWER':'avoidance'}, {'LOWER':'directive'}],
            [{'LOWER':'atad'}])

cfc_matcher = spacy.matcher.Matcher(nlp.vocab)
cfc_matcher.add('Controlled Foreign Corporation', None, 
            [{'LOWER':'controlled'}, {'LOWER':'foreign'}, {'LOWER':'corporation'}])

corporate_tax_matcher = spacy.matcher.Matcher(nlp.vocab)
corporate_tax_matcher.add('Corporate Tax', None, 
            [{'LOWER':'corporate'}, {'LOWER':'tax'}],
            [{'LOWER':'multi-nationals'}],
            [{'LOWER':'corporate'}, {'LOWER':'income'},{'LOWER':'tax'}])

cbcr_matcher = spacy.matcher.Matcher(nlp.vocab)
cbcr_matcher.add('Country-by-Country Reporting', None, 
            [{'LOWER':'country-by-country'}, {'LOWER':'reporting'}],
            [{'LOWER':'country-by-country'}],
            [{'LOWER':'CbC'}],
            [{'LOWER':'CbCR'}])

digital_tax_matcher = spacy.matcher.Matcher(nlp.vocab)
digital_tax_matcher.add('Digital Tax', None, 
            [{'LOWER':'digital'}, {'LOWER':'tax'}],
            [{'LOWER':'digital'}, {'LOWER':'services'},{'LOWER':'tax'}],
            [{'LOWER':'digitalised'}],
            [{'LOWER':'digitalized'}],
            [{'LOWER':'making'}, {'LOWER':'tax'}, {'LOWER':'digital'}],
            [{'LOWER':'"web'}, {'LOWER':'tax"'}, {'LOWER':'on'}, {'LOWER':'digital'},{'LOWER':'services'}],
            [{'LOWER':'tax'}, {'LOWER':'on'}, {'LOWER':'digital'}, {'LOWER':'activity'}])

double_tax_treaty_matcher = spacy.matcher.Matcher(nlp.vocab)
double_tax_treaty_matcher.add('Double Tax Treaty', None, 
            [{'LOWER':'double'}, {'LOWER':'tax'}, {'LOWER':'treaty'}],
            [{'LOWER': 'double'}, {'LOWER': 'tax'}],
            [{'LOWER': 'double'}, {'LOWER': 'taxation'}])

dac6_matcher = spacy.matcher.Matcher(nlp.vocab)
dac6_matcher.add('EU Mandatory Disclosure Directive (DAC6)', None, 
            [{'LOWER':'eu'}, {'LOWER':'mandatory'}, {'LOWER':'disclosure'}, {'LOWER':'directive'}],
            [{'LOWER':'dac6'}],
            [{'LOWER': 'directive'}, {'LOWER':'on'}, {'LOWER': 'administrative'}, {'LOWER':'cooperation'}],
            [{'LOWER': 'dac 6'}])

economic_substance_matcher = spacy.matcher.Matcher(nlp.vocab)
economic_substance_matcher.add('Economic substance', None, 
            [{'LOWER':'economic'}, {'LOWER':'substance'}],
            [{'LOWER':'business'},{'LOWER':'substance'}],
            [{'LOWER':'substance'}],
            [{'LOWER':'substantive'},])

gaar_matcher = spacy.matcher.Matcher(nlp.vocab)
gaar_matcher.add('General anti-avoidance rule (GAAR)', None, 
            [{'LOWER':'general'}, {'LOWER':'anti-avoidance'},{'LOWER':'rule'}],
            [{'LOWER':'gaar'}],
            [{'LOWER':'general'},{'LOWER':'anti-abuse'},{'LOWER':'rules'}],
            [{'LOWER':'anti-avoidance'},{'LOWER':'rules'}])

gst_matcher = spacy.matcher.Matcher(nlp.vocab)
gst_matcher.add('Goods and services tax (GST)', None, 
            [{'LOWER':'goods'}, {'LOWER':'and'},{'LOWER':'services'}, {'LOWER':'tax'}],
            [{'LOWER':'gst'}])

interest_deductibility_matcher = spacy.matcher.Matcher(nlp.vocab)
interest_deductibility_matcher.add('Interest deductibility', None, 
            [{'LOWER':'interest'}, {'LOWER':'deductibility'}],
            [{'LOWER':'interest'}, {'LOWER':'deductions'}],
            [{'LOWER':'deductibility'},{'LOWER':'of'},{'LOWER':'interest'}],
            [{'LOWER':'interest'},{'LOWER':'limitation'}],
            [{'LOWER':'borrowing'},{'LOWER':'costs'}],
            [{'LOWER':'interest'},{'LOWER':'expenses'}],
            [{'LOWER':'deduction'},{'LOWER':'of'},{'LOWER':'interest'}])

local_file_matcher = spacy.matcher.Matcher(nlp.vocab)
local_file_matcher.add('Local file', None, 
            [{'LOWER':'local'}, {'LOWER':'file'}],
            [{'LOWER':'local'},{'LOWER':'filing'}])

mandatory_disclosure_rules_matcher = spacy.matcher.Matcher(nlp.vocab)
mandatory_disclosure_rules_matcher.add('Mandatory disclosure rules', None, 
            [{'LOWER':'mandatory'}, {'LOWER':'disclosure'}, {'LOWER':'rules'}],
            [{'LOWER':'disclosure'}, {'LOWER':'rules'}],
            [{'LOWER':'disclosures'}])

master_file_matcher = spacy.matcher.Matcher(nlp.vocab)
master_file_matcher.add('Master file', None, 
            [{'LOWER':'master'}, {'LOWER':'file'}],
            [{'LOWER':'master'}, {'LOWER':'filing'}])

mli_matcher = spacy.matcher.Matcher(nlp.vocab)
mli_matcher.add('Multilateral Instrument (MLI)', None, 
            [{'LOWER':'multilateral'}, {'LOWER':'instrument'}],
            [{'TEXT':'MLI'}])

map_matcher = spacy.matcher.Matcher(nlp.vocab)
map_matcher.add('Mutual agreement procedure', None, 
            [{'LOWER':'mutual'}, {'LOWER':'agreement'}, {'LOWER':'procedure'}],
            [{'LOWER': 'map'}])

pe_matcher = spacy.matcher.Matcher(nlp.vocab)
pe_matcher.add('Permanent Establishment (PE)', None, 
            [{'LOWER':'permanent'}, {'LOWER':'establishment'}],
            [{'TEXT':'PE'}])

principal_purpose_matcher = spacy.matcher.Matcher(nlp.vocab)
principal_purpose_matcher.add('Principal purpose test', None, 
            [{'LOWER':'principal'}, {'LOWER':'purpose'}, {'LOWER':'test'}],
            [{'LOWER':'principal'},{'LOWER':'purpose'}])

section892_matcher = spacy.matcher.Matcher(nlp.vocab)
section892_matcher.add('Section 892', None, 
            [{'LOWER':'section'}, {'LOWER':'892'}])

sovereign_immunity_matcher = spacy.matcher.Matcher(nlp.vocab)
sovereign_immunity_matcher.add('Sovereign Immunity', None, 
            [{'LOWER':'sovereign'}, {'LOWER':'immunity'}])

tax_governance_framework_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_governance_framework_matcher.add('Tax Governance Framework', None, 
            [{'LOWER':'tax'}, {'LOWER':'governance'}, {'LOWER':'framework'}],
            [{'LOWER':'regulatory'},{'LOWER':'framework'}],
            [{'LOWER':'tax'},{'LOWER':'framework'}])

tax_audit_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_audit_matcher.add('Tax audit', None, 
            [{'LOWER':'tax'}, {'LOWER':'audit'}],
            [{'LOWER':'audit'},{'LOWER':'procedure'}],
            [{'LOWER':'revenue'},{'LOWER':'audit'}],
            [{'LOWER':'audit'}])

tax_compliance_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_compliance_matcher.add('Tax compliance', None, 
            [{'LOWER':'tax'}, {'LOWER':'compliance'}])

tax_dispute_matcher = spacy.matcher.Matcher(nlp.vocab)
tax_dispute_matcher.add('Tax dispute', None, 
            [{'LOWER':'tax'}, {'LOWER':'dispute'}],
            [{'LOWER':'dispute'}, {'LOWER':'resolution'}],
            [{'LOWER':'action'}, {'LOWER':14}],
            [{'LOWER':'disputes'}],
            [{'LOWER':'tax'}, {'LOWER':'in'}, {'LOWER':'dispute'}])

vat_matcher = spacy.matcher.Matcher(nlp.vocab)
vat_matcher.add('Value-added tax (VAT)', None, 
            [{'LOWER':'value-added'}, {'LOWER':'tax'}])

witholding_tax_matcher = spacy.matcher.Matcher(nlp.vocab)
witholding_tax_matcher.add('Withholding Tax', None,
            [{'LOWER':'withholding'}, {'LOWER':'tax'}])



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
#df=pd.read_csv('/data/home/gicpoc/gic/data_sets/test1.csv')
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
matcher_list = [management_tax_function_matcher, atad_matcher, cfc_matcher, corporate_tax_matcher, cbcr_matcher, digital_tax_matcher,
                double_tax_treaty_matcher, dac6_matcher, economic_substance_matcher, gaar_matcher, gst_matcher,
                interest_deductibility_matcher, local_file_matcher, mandatory_disclosure_rules_matcher, master_file_matcher,
                mli_matcher, map_matcher, pe_matcher, principal_purpose_matcher, section892_matcher, sovereign_immunity_matcher,
                tax_governance_framework_matcher, tax_audit_matcher, tax_compliance_matcher, tax_dispute_matcher,
                vat_matcher, witholding_tax_matcher]

columns=['(Management of a) Tax function',
    'Anti-Tax Avoidance Directive (ATAD)',
    'Controlled Foreign Corporation',
    'Corporate Tax',
    'Country-by-Country Reporting',
    'Digital Tax',
    'Double Tax Treaty',
    'EU Mandatory Disclosure Directive (DAC6)',
    'Economic substance',
    'General anti-avoidance rule (GAAR)',
    'Goods and services tax (GST)',
    'Interest deductibility',
    'Local file',
    'Mandatory disclosure rules',
    'Master file',
    'Multilateral Instrument (MLI)',
    'Mutual agreement procedure',
    'Permanent Establishment (PE)',
    'Principal purpose test',
    'Section 892',
    'Sovereign Immunity',
    'Tax Governance Framework',
    'Tax audit',
    'Tax compliance',
    'Tax dispute',
    'Value-added tax (VAT)',
    'Withholding Tax']

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
#subcats.to_pickle('/data/home/gicpoc/gic/tmp_poc/v1_matcher_27_test.pkl')
subcats.to_pickle(sys.argv[2])

#for matcher, category in zip(matcher_list, columns): 
#    got_matches = get_matches(matcher, tokenized_texts_valid)
#    match_count = [Counter(doc) for doc in got_matches]
#    print(len(match_count))
#    subcats_valid[category]=match_count
#subcats_valid['filename']=df_valid['filename']
#subcats_valid.to_pickle('/data/home/gicpoc/gic/tmp_poc/v1_matcher_27_validate.pkl')

def counter2ls(df_column):
    ls_col = [list(doc.items()) for doc in df_column]
    return ls_col
