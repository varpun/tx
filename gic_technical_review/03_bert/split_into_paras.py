import pandas as pd
import os
import pathlib
import numpy as np


## This script will look through the data folder in your current working directory, look for any test files in csv form, and split the text into paragraphs
## for input into BERT. If the split files already exist, it will exit


# Helper Functions
def explode(df, lst_cols, fill_value='', preserve_index=False):
  
  """
  Dataframe + ColumnContainingListsToSplitInto Rows + (fillValie + Boolean) - > Dataframe
  
  Takes in a dataframe and a list of the columns that have the lists you want to split, then splits into 
  separate rows
  """
  # make sure `lst_cols` is list-alike
  if (lst_cols is not None
      and len(lst_cols) > 0
      and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
      lst_cols = [lst_cols]
  # all columns except `lst_cols`
  idx_cols = df.columns.difference(lst_cols)
  # calculate lengths of lists
  lens = df[lst_cols[0]].str.len()
  # preserve original index values    
  idx = np.repeat(df.index.values, lens)
  # create "exploded" DF
  res = (pd.DataFrame({
              col:np.repeat(df[col].values, lens)
              for col in idx_cols},
              index=idx)
           .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                          for col in lst_cols}))
  # append those rows that have empty lists
  if (lens == 0).any():
      # at least one list in cells is empty
      res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                .fillna(fill_value))
  # revert the original index order
  res = res.sort_index()
  # reset index if requested
  if not preserve_index:        
      res = res.reset_index(drop=True)
  return res

def split_text_into_paras_test(df):
  """
  DataFrame-> DataFrame
  
  Takes a DataFrame with "text" column and splits it into different paragraphs depending on whether it has pdf in its "Filename"
  """
  new_df = df.copy()
  for index, row, in df.iterrows():
    new_df.at[index, "text"] = row["text"].split("\n")
  return new_df

def strip_list_noempty(mylist):
  """
  ListOfStrings->ListOfStrings
  
  Takes a list of strings and filters it so that it will drop all whitespace strings
  """
  newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
  return [item for item in newlist if item or not hasattr(item, 'strip')]

def remove_whitespace_str_from_df(df):
  """
  DataFrame-> DataFrame
  
  Removes empty strings or whitespace strings like "", " ", "\n" etc from ListOfStrings in "text" series of DF
  """
  new_df = df.copy()
  for index, row in df.iterrows():
    new_df.at[index, "text"] = strip_list_noempty(row["text"])
  return new_df



# Main program

cwd = pathlib.Path.cwd()
parent = cwd.parent
test_folder = parent / "data" / "test"
test_folder=test_folder.glob('**/*')

# data_folder=data_folder.glob('**/*')

files = [x for x in test_folder if x.is_file()]
test_files = list(filter(lambda x: "test" in x.stem, files))
exploded_files = list(filter(lambda x: "exploded" in x.stem, test_files))
# exploded_files = list(filter(lambda x: "exploded" in x.stem, test_files))
non_exploded = list(set(test_files).difference(exploded_files))

if len(exploded_files) == len(non_exploded):
    print("All test files have been split into paragraphs.")
    
else:
    for test_file in test_files:
        print(f"Splitting {test_file.stem}")
        test_df = pd.read_csv(test_file, usecols=["Index","Reference","text"])
        test_df = test_df.drop(columns=["Index"],axis=1)
        exploded_test = explode(remove_whitespace_str_from_df(split_text_into_paras_test(test_df)), ["text"])
        exploded_test = exploded_test[["Reference","text"]]
        exploded_test.to_csv(test_file.parent / (test_file.stem+"_BERT_exploded.csv"), index=False)
    print("All test files have been split into paragraphs for input into BERT.")



