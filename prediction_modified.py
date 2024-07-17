from transformers import AutoTokenizer
import wandb
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from utils import check_max_token_length
from dataset_and_model import Dataset, Model

torch.cuda.empty_cache()
emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M']

save_dir = r'/content/drive/MyDrive/Personal/publications/annotations/models/political_emotion_model_pl_roberta_modified'
dropout = 0.6
hidden_dim = 768
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = r'D:\PycharmProjects\roberta\roberta_base_transformers'
tokenizer = AutoTokenizer.from_pretrained(model)

# max_len = check_max_token_length(tokenizer)

model = Model(model_dir = model, metric_names = emotion_columns, dropout = dropout, hidden_dim = hidden_dim)

model.load_state_dict(torch.load('models/political_emotion_model_pl_roberta_modified'))
model.to(device)
model.eval()


def predict_emotions(texts):
    # Ensure the input is in list format
    if isinstance(texts, str):
        texts = [texts]  # Wrap single string in a list

    # Tokenize all text snippets at once
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Move tensors to the appropriate device
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    # Disable gradient calculations
    with torch.no_grad():
        # Forward pass, get model predictions for the entire batch
        outputs = model(input_ids, attention_mask)

    # Prepare results for each text snippet
    # Process each tensor in the output list
    results = []
    for i in range(len(texts)):  # Iterate over each text
        # Extract the score for each emotion for the current text
        text_results = {emotion: outputs[j][i].cpu().float().item() for j, emotion in enumerate(emotion_columns)}
        results.append(text_results)

    # Return a single result if a single text was provided, else return the list of results
    return results[0] if len(results) == 1 else results
##################
# POLITICAL COMPASS
##################
from mypolitics import sentences

df = pd.DataFrame(sentences, columns=['item'])

results = predict_emotions(sentences)



emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M']
for i, emotion in enumerate(emotion_columns):
    df[emotion] = [result[emotion] for result in results]


# save
df.to_csv('political_emotion_results_modified.csv', index=False)

# to excel
df.to_excel('political_emotion_results_modified.xlsx', index=False)

# sort by Valence_M
df.sort_values(by='Valence_M', ascending=False, inplace=True)

#############
# POLITICIANS
#############

politicians = pd.read_excel('politicians.xlsx')

results = predict_emotions(list(politicians['Politician'].values))

emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M']
for i, emotion in enumerate(emotion_columns):
    politicians[emotion + '_modified'] = [result[emotion] for result in results]

# import correlation
from scipy.stats import pearsonr

# correlate
corr = pearsonr(politicians['Valence_M'], politicians['Valence_M_modified'])


# save
politicians.to_csv('politicians.csv', index=False)

# to excel
politicians.to_excel('politicians.xlsx', index=False)

# sort by Valence_M
politicians.sort_values(by='Valence_M', ascending=False, inplace=True)

###################
# NEUTRAL SENTENCES
###################

politicians = pd.read_excel('politicians.xlsx')

names = list(politicians['Politician'].values)

from neutral_sentences import neutral_sentences


emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M']

rows = []
for politician in names:
    for sentence in neutral_sentences:
        sentence = sentence.replace('[Name]', politician)
        score = predict_emotions(sentence)
        score['Politician'] = politician
        score['Sentence'] = sentence
        rows.append(score)

# tp df
df = pd.DataFrame(rows)

# save
df.to_csv('neutral_sentences_results_modified.csv', index=False)

# load
neutral = pd.read_csv('neutral_sentences_results.csv')

# groupby Politician and mean
del neutral['Sentence']
neutral = neutral.groupby('Politician').mean().reset_index()

#####################
# POLITICAL SENTENCES
#####################

politicians = pd.read_excel('politicians.xlsx')

# load

names = list(politicians['Politician'].values)

from neutral_sentences import political_sentences


emotion_columns = ['Happiness_M', 'Sadness_M', 'Anger_M', 'Disgust_M', 'Fear_M', 'Pride_M', 'Valence_M', 'Arousal_M']

rows = []
for politician in names:
    for sentence in political_sentences:
        sentence = sentence.replace('[Name]', politician)
        score = predict_emotions(sentence)
        score['Politician'] = politician
        score['Sentence'] = sentence
        rows.append(score)

# tp df
df = pd.DataFrame(rows)

# save
df.to_csv('political_sentences_results_modified.csv', index=False)


del df['Sentence']
df = df.groupby('Politician').mean().reset_index()
