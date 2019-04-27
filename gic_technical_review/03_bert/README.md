How to get BERT Predictions for each sub-category as Probabilities

1. Run split_into_paras.py. This will split the text in the test files in /data/test/ into paragraphs to input into BERT
(We have already ran the script and the outputs are available in the /data/test/ folder as ..._BERT_exploded_
2. Run bert_predictory.py This will read the files with split paragraphs and output predictions
(We have already ran the predictor script and the outputs are available in the /data/results/ folder)
