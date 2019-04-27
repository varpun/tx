# Exploratory MVP to leverage NLP for Tax Notifications

AI Singapore and the GIC Tax Team recently concluded a Proof-of-concept exercise exploring the use of machine learning extract relevant tax subcategories and countries from notifications the tax team receives. This code repo contains the POC test data, scripts and other artefacts needed to reproduce the POC results 

# Instructions 
1. Run the Rules Matcher to extract exact subcategory mentions from raw text. We have pre-executed the script and the results can be found as `./data/v1_matcher_27_test.pkl` and `./data/v1_matcher_test.pkl`

`cd subcat_matcher`   
 
`bash generate_matches.sh`

2. Run `02_prob_gen.py` to generate multilabel tax-category predictions. This script should be run once for every test set ie test1.csv, test2.csv and test3.csv (We have pre-run the script and the output can be found in `data/test1_probabilities.csv`, `data/test2_probabilities.csv`,`data/test3_probabilities.csv`)

`python 02_prob_gen.py`

3. Run BERT model 

Run `03_bert/split_into_paras.py`. This will split the text in the test files in /data/test/ into paragraphs to input into BERT
(We have already ran the script and the outputs are available in the `/data/test/ folder as ..._BERT_exploded_`

Run `03_bert/bert_predictor.py` This will read the files with split paragraphs and output predictions
(We have already ran the predictor script and the outputs are available in the /data/results/ folder)

`python 03_bert/split_into_paras.py`
`python 03_bert/bert_predictor.py`

* note that you may have to run `pip install -r 03_bert/requirements.txt` to install custom dependencies for BERT 

4. Run `04_country_predictor.py` to get country predictions

`python 04_country_predictor.py` 

(We have already run the script and the results can be found as `data/test1_country_predictions.csv`,`data/test2_country_predictions.csv`,`data/test3_country_predictions.csv`)

5. See `05_poc_analysis.ipynb` for code on how the results of the POC was obtained 


# Methods 

## Background
The objective is to use natural language processing to extract key concepts from the notifications the tax team receives. The tax team can review the draft prepared by the program and update it/tailor it to GIC’s circumstances. This will enable the tax team to use their time on more value added tasks and improve the frequency and coverage of the team.

## Data Collection 
Training Data: Training data was obtained through web scraping. Each tax subcategory was put into a search engine, and if an exact mention of that tax subcategory appeared in the header of the search results, we added the search result URL to our dataset and tagged it by the tax subcategory entered into the search engine. The tax team provided additional examples for subcategories that were underrepresented after web scraping. 

Validation Data: Validation data consists of 193 tax notifications previously received by the tax team via their sources. These notifications were manually labelled accordingly to the 27 tax subcategories by 3 tax analysts. 

Test Data: Test data consists of 60 tax notifications previously received by the tax team via their sources. These notifications were manually labelled accordingly to the 27 tax subcategories by a set of 3 tax analysts separate from the analysts who labelled the validation data. 

## Model  
Our tax subcategory predictor is an ensemble of a BERT classifier and a tf-idf ensemble classifier (Random Forests, SVM, Gradient Boosting, Logistic Regression). 
The model's output is a weighted average of the BERT classifier (50%) and the tf-idf ensemble classifier (50%)

For country predictions, we augmented the spaCy libraries Named Entity Recognition model with a set of rules to tailor the model for our use case. Eg. mentions of HMRC refer to the UK. We then extracted spaCy entities tagged as geo-political entities (GPE) and filtered the entities to include only our countries of interest. 

## Results Analysis 
Analysis of results can be found in the slides POC_results_24th_April_2019.pptx
