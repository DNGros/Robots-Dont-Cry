# Features Analysis
import sys

from joblib import Memory

sys.path.append("../")
from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results
from real_robots_dont_cry.gensurvey import QuestionTopic
from real_robots_dont_cry.join_results import get_joined_results, DEMOGRAPHIC_QUESTION_TO_COL
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy import stats
import nltk
import language_tool_python
from profanity_check import predict, predict_prob
from tqdm import tqdm
tqdm.pandas()

# grammar tool
my_tool = language_tool_python.LanguageTool('en-US')

cachedir = 'features_cache'
diskcache = Memory(cachedir, verbose=2)



def get_sentiment(row):
    return TextBlob(row['turn_b']).sentiment.polarity

def get_length(row):
    words = nltk.word_tokenize(row['turn_b'])
    return len(words)

def get_profanity(row):
    return predict_prob([row['turn_b']]).tolist()[0]

def get_grammar(row):
    my_matches = my_tool.check(row['turn_b'])
    return len(my_matches)


@diskcache.cache
def get_features_df():
    df = get_filtered_joined_results()
    df = df.reset_index()  # make sure indexes pair with number of rows
    print("analysing sentiment...")
    df['sentiment'] = df.progress_apply(get_sentiment, axis=1)
    print("analysing length...")
    df['length'] = df.progress_apply(get_length, axis=1)
    print("analysing profanity...")
    df['profanity'] = df.progress_apply(get_profanity, axis=1)
    print("analysing grammar...")
    df['grammar'] = df.progress_apply(get_grammar, axis=1)
    return df