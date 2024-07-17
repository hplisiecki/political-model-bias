import pandas as pd
import os

dir = 'data/subsets_to_check'

collate_df = []
lengths = []
for file in os.listdir(dir):
    df = pd.read_csv(os.path.join(dir, file))
    collate_df.append(df)
    lengths.append(len(df))



collate_df = pd.concat(collate_df, axis = 0)

indexes_to_delete = collate_df['Unnamed: 0'].tolist()

train_set = pd.read_csv('data/train_set.csv')

train_set = train_set.drop(indexes_to_delete, axis = 0)

# save
train_set.to_csv('data/modified_train_set.csv', index = False)