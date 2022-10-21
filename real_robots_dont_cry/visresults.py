import sys
import textwrap
from pathlib import Path
import pandas as pd

from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, is_a_quality_check_row

cur_file = Path(__file__).parent.absolute()
# Why not in the path on laptop???
if cur_file.parent not in sys.path:
    sys.path.insert(0, str(cur_file.parent))

import seaborn as sns
from matplotlib import pyplot as plt
from real_robots_dont_cry.join_results import get_joined_results


def print_sample(df):
    # iterate over sample of each row as a dict
    for row in df.sample(10).to_dict(orient='records'):
        print("    Human>", row['turn_a'])
        print(f"R {row['bot_desc_cat']}> {row['turn_b']}")
        print(row['question_text'])
        print(row['ans'])
        print("-" * 20)


def print_all(df):
    """For each `text_hash` print all the `ans` grouped by `question_topic`"""
    df = df[df.is_duplicate == False]
    for text_hash, group in df.groupby('text_hash'):
        print("-" * 20)
        # Get the turn_a and turn_b for the group
        turn_a = group.iloc[0]['turn_a']
        turn_b = group.iloc[0]['turn_b']
        # Print out the dialogue
        print(group.iloc[0]['dataset_src'])
        def wrap(text):
            return '\n'.join(textwrap.wrap(text, width=60, subsequent_indent='       '))
        print(f"Human> {wrap(turn_a)}")
        print(f" Resp> {wrap(turn_b)}")
        # for (question_text, question_topic), group_ans in group.groupby(['question_text', 'question_topic']):
        for resp_cat, group_ans in reversed(list(group.groupby(['resp_cat']))):
            # print(f"{question_text} ({bot_desc_cat})")
            clean_resp_cat = {
                'truthful_r-humanoid': 'Humanoid Possible',
                'truthful_human': 'Actual Human Possible',
                'truthful_r-chatbot': 'Chatbot Possible',
                'comfort_r-humanoid': 'Humanoid Comfortable',
                'comfort_human': 'Actual Human Comfortable',
                'comfort_r-chatbot': 'Chatbot Comfortable',
            }.get(resp_cat, resp_cat)
            print(f"{clean_resp_cat}: "
                  f'{group_ans["ans"].mean():.2f} '
                  f'({",".join(map(str, group_ans["ans"].values))})')


def print_surveys_total_stats(df):
    # get all unique values of `q_kind` in df
    unique_q_kinds = df.q_kind.unique()
    df['question_topic'] = df['question_topic'].map(lambda x: {
        'human-truthful': 'human-possible',
        'robot-truthful': 'robot-possible',
    }.get(x, x))
    is_human_vals = df[df.is_human == True]
    # print the ans mean and count
    print("Human")
    print(is_human_vals.groupby('question_topic').ans.agg({'mean', 'count'}))
    is_robot_vals = df[df.is_human == False]
    print("--")
    print("Any Robot")
    print(is_robot_vals.groupby('question_topic').ans.agg({'mean', 'count'}))
    is_chatbot_vals = df[(df.is_human == False) & (df.bot_desc_cat == 'chatbot')]
    print("--")
    print("Chatbot")
    print(is_chatbot_vals.groupby('question_topic').ans.agg({'mean', 'count'}))
    is_humanoid_vals = df[(df.is_human == False) & (df.bot_desc_cat == 'humanoid')]
    print("--")
    print("Humanoid")
    print(is_humanoid_vals.groupby('question_topic').ans.agg({'mean', 'count'}))


def main():
    df = get_filtered_joined_results()
    #df = get_joined_results()
    #df = df[df['dataset_src'] == 'msc']
    #df = df[~df['is_human'] & ~df['is_duplicate'] & ~is_a_quality_check_row(df)]
    #print(df['task_id'].unique())
    print(df.groupby(
        # ['question_topic', 'bot_desc_cat']#, 'dataset_src']
        ['text_hash', 'question_topic']
    ).count())
    print_all(df)
    g = sns.FacetGrid(df, col="question_topic")
    g.map(sns.countplot, "ans")
    plt.show()
    print_surveys_total_stats(df)
    # Print counts by dataset_src
    print(df.groupby('dataset_src').nunique())
    pd.set_option('display.max_columns', 5)
    print(df.groupby(['dataset_src', 'resp_cat']).agg({'mean', 'count'}))
    print(len(df))


if __name__ == "__main__":
    main()
