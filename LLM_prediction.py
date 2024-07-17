from langtorch.tt import Activation, TextModule
from langtorch import TextTensor
from langtorch.api import auth
from tqdm import tqdm
import pandas as pd
import numpy as np
api_file = r"C:\Users\hplis\OneDrive\Desktop\PHD\OPENAI\api_keys.json"

auth(api_file)


emotions = ['valence', 'arousal']
emotion_columns = ['norm_Valence_M', 'norm_Arousal_M']

def multiple_shot(texts, examples = [], values = [], debug = False, model = "gpt-3.5-turbo-1106"):


    query = (
        f'Jaki znak emocji wyczytujesz w poniższym tekście? Odpowiedz używając 10 stopniowej skali, '
        f'gdzie 1 - obecna jest negatywna emocja a 10 - obecna jest pozytywna emocja. Odpowiadaj za pomocą '
        f'pojedynczego numeru. ')

    collated = []
    idx = 0




    for idx, tuple in enumerate(zip(examples, values)):
        query += f' Tekst {idx + 1}: """{tuple[0]}""" Twoja odpowiedź: """{tuple[1]}""" ###'

    if debug:
        print(query)
        return

    next_text_idx = idx


    annotation_module = Activation(model=model, system_message=query, T=0, max_characters=12)

    for text in tqdm(texts):
        annotation = annotation_module(TextTensor(f' Tekst {next_text_idx + 2}: """{text}""", Twoja odpowiedź: '))
        collated.append(str(annotation))

    return collated

# POLITICIANS
politicians = pd.read_excel('politicians.xlsx')

results = multiple_shot(list(politicians['Politician'].values))

politicians['LLM'] = results

# save
politicians.to_csv('politicians.csv', index=False)
# to excel
politicians.to_excel('politicians.xlsx', index=False)

############
# SENTENCES
############


politicians = pd.read_excel('politicians.xlsx')

names = list(politicians['Politician'].values)

from neutral_sentences import neutral_sentences


emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M']

valence_list = []
sentences_list = []
politicians_list = []
for politician in names:
    for sentence in neutral_sentences:
        sentence = sentence.replace('[Name]', politician)
        score = multiple_shot([sentence])
        valence_list.append(score[0])
        sentences_list.append(sentence)
        politicians_list.append(politician)

# to dataframe
df = pd.DataFrame({'Politician': politicians_list, 'Sentence': sentences_list, 'LLM': valence_list})
# save
df.to_csv('neutral_sentences_results_LLM.csv', index=False)


# POLITICIANS
politicians = pd.read_excel('politicians.xlsx')

results = multiple_shot(list(politicians['Politician'].values),  model = 'gpt-4')

politicians['LLMgpt4'] = results

# save
politicians.to_csv('politicians.csv', index=False)

# to excel
politicians.to_excel('politicians.xlsx', index=False)

