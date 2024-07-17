import pandas as pd

train_set = pd.read_csv('data/train_set.csv')

politicians = pd.read_excel('politicians.xlsx')

names = list(politicians['Politician'].values)

names = [n.split(' ')[1] for n in names]

texts = list(train_set['text'].values)


snippets = {}
for name in names:
    snippets[name] = [text for text in texts if name in text]