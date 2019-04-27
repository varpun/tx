#!/bin/bash
FILEPATH='./data/test1.csv'

OUTPATH1='./data/v1_matcher_test1.pkl'
OUTPATH2='./data/v1_matcher_27_test1.pkl'

python ./src/subcat_token_matcher.py $FILEPATH $OUTPATH1
python ./src/subcat_token_matcher_27cats.py $FILEPATH $OUTPATH2

FILEPATH='./data/test2.csv'

OUTPATH1='./data/v1_matcher_test2.pkl'
OUTPATH2='./data/v1_matcher_27_test2.pkl'

python ./src/subcat_token_matcher.py $FILEPATH $OUTPATH1
python ./src/subcat_token_matcher_27cats.py $FILEPATH $OUTPATH2

FILEPATH='./data/test3.csv'

OUTPATH1='./data/v1_matcher_test3.pkl'
OUTPATH2='./data/v1_matcher_27_test3.pkl'

python ./src/subcat_token_matcher.py $FILEPATH $OUTPATH1
python ./src/subcat_token_matcher_27cats.py $FILEPATH $OUTPATH2

echo "DONE"
