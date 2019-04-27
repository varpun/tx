

def counter(col)
    c = [Counter(entry) for entry in col]
    return c 

def counter_set(col):
    entries = [set(entry) for entry in col]
    return entries 

subcat_sets = subcats.iloc[:, 0:10].apply(counter_set, axis=0)

def set2ls(col):
    ls  = [list(entry) for entry in col]
    return ls 
