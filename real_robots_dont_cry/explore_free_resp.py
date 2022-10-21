from pathlib import Path

import numpy as np
import pandas as pd
import tabulate

from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results
from pprint import pprint

from real_robots_dont_cry.gensurvey import QuestionTopic, QuestionText
from real_robots_dont_cry.jinjatoy.render_jinja import FRIENDLY_NAME_MAP, FRIENDLY_NAME_MAP_SHORTER
from real_robots_dont_cry.join_results import get_pilot_3, get_joined_results
from real_robots_dont_cry.new_json_filter_by_qual_check import get_passing_non_passing_df
from real_robots_dont_cry.visresults import print_all
import colorama

from util.sampling import deterministic_hash
from util.util import flatten_list

cur_file = Path(__file__).parent.absolute()


def print_free_resp():
    df, non_passing_df, passing_df = get_passing_non_passing_df(
        consider_free_resp=True,
    )
    df = passing_df
    #print(f"{len(df)=}")
    #df = get_filtered_joined_results()
    #print(f"{len(df)=}")
    has_explan = df[df.user_explanation.notnull()]
    print_count = 0
    print(len(has_explan))
    for (text_hash, task_id), group in has_explan.groupby(['text_hash', 'task_id']):
        #print(f"{text_hash=} {task_id=}")
        if group.ans.min() >= 4:
            continue
        #print_all(group)
        print("EXPLANATION:")
        pprint(group.user_explanation.iloc[0])
        print_count += 1
    print(f"Printed {print_count} explanations")
    pass


def dump_explans_to_csv():
    df, non_passing_df, passing_df = get_passing_non_passing_df(
        consider_free_resp=True,
    )
    df = passing_df
    has_explan = df[df.user_explanation.notnull()]
    print_count = 0
    print(len(has_explan))
    vals = []
    print(df.keys())
    for (text_hash, task_id), group in has_explan.groupby(['text_hash', 'task_id']):
        if group.ans.min() >= 4:
            continue
        vals.append({
            "text_hash": text_hash,
            "task_id": task_id,
            #"dataset_src": group.dataset_src.iloc[0],
            "turn_a": group.turn_a.iloc[0],
            "turn_b": group.turn_b.iloc[0],
            "bot_desc_cat": group.bot_desc_cat.iloc[0],
            **{
                row['question_topic']: row['ans']
                for _, row in group.iterrows()
            },
            "explanation": group.user_explanation.iloc[0],
        })
        print_count += 1

    pd.DataFrame(vals).to_csv(cur_file / "generations/user_explans_v1.csv")
    print(f"Saved {print_count} explanations")
    pass


def sort_filled_explans():
    df = pd.read_csv(cur_file / "generations/user_explans_v1_fill.csv")
    print(df)
    print("num double", len(df[(df['Has Label'] > 1) & (df['Generic / No Explan']).isnull()]))
    fill_cols = df.columns[12:]
    print(f"Num fill cols: {len(fill_cols)}")
    fill_cols = sorted(fill_cols, key=lambda col: df[col].sum(), reverse=False)
    for col in fill_cols:
        print(colorama.Fore.BLUE + col + colorama.Fore.RESET)
        print(df[col].sum())
        if df[col].sum() < 25:
            for i, row in df[df[col] == 1].iterrows():
                print(row.explanation)


def make_table():
    df = pd.read_csv(cur_file / "generations/user_explans_v2_fill.csv")
    df['text_hash'] = df.apply(
        lambda row: deterministic_hash((row['turn_a'], row['turn_b']), seed=0),
        axis=1
    )
    orig_data = get_joined_results()
    orig_data = orig_data[orig_data.user_explanation.notnull()]
    orig_data = orig_data[orig_data.question_topic == QuestionTopic.ROBOT_POSSIBLE.value]
    fill_cols = df.columns[12:]
    fill_cols = sorted(fill_cols, key=lambda col: df[col].sum(), reverse=True)
    table_rows = []
    #add_tfidf_vecs(df, 'explanation')


    # TFidf
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import text
    question_words = flatten_list([
        q.value.lower().split()
        for q in QuestionText
    ])
    question_words = list(set(question_words))
    question_words.extend(
        ['robot', 'humanoid', 'chatbot',
         'robots', 'humans', 'bots', "uncomfortable",
         "impossible", "doesn", "didn", 'untruthful']
    )
    print(question_words)
    my_stop_words = text.ENGLISH_STOP_WORDS.union(
        question_words
    )
    # Use NLTK's PorterStemmer
    from nltk.stem.porter import PorterStemmer
    porter_stemmer = PorterStemmer()
    import re
    def stemming_tokenizer(str_input):
        str_input = str_input.lower()
        words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
        words = [porter_stemmer.stem(word) for word in words]
        return words
    vectorizer = TfidfVectorizer(
        stop_words=my_stop_words,
        binary=True,
        norm='l1',
        #tokenizer=stemming_tokenizer,
    )
    fit_vectorizer = vectorizer.fit(df['explanation'])

    # Do cols
    for col in fill_cols:
        if col in (
            "Rating Inconsistant w/ Explan",
            "Multiturn",
            "Wrongly Aggregate",
            "Incoherent Explan",
        ):
            continue
        match_rows = df[df[col] == 1]
        assert len(match_rows) >= 1
        # get the datatype for the text_hash column
        assert orig_data.text_hash.dtype == match_rows.text_hash.dtype, \
            f"{orig_data.text_hash.dtype=} {match_rows.text_hash.dtype=}"
        assert orig_data.task_id.dtype == match_rows.task_id.dtype
        # join in original data by text_hash and task_id
        old_len = len(match_rows)
        match_rows = match_rows.merge(
            orig_data,
            on=['text_hash', 'task_id'],
            how='left',
            validate="one_to_one",
        )
        word_vecs = fit_vectorizer.transform(match_rows['explanation']).todense()
        print(word_vecs.shape)
        word_vecs = word_vecs.sum(axis=0).reshape((len(vectorizer.get_feature_names()),))
        word_vecs = np.squeeze(np.asarray(word_vecs))
        print(type(word_vecs))
        print(word_vecs.shape)
        max_indicies = np.argpartition(word_vecs, -3)[-3:]
        print(max_indicies)
        max_words = [vectorizer.get_feature_names()[i] for i in max_indicies]
        assert len(match_rows) >= 1
        assert len(match_rows) == old_len, f"{len(match_rows)=} {old_len=}"
        match_datasets = match_rows.groupby('dataset_src').count()
        match_datasets = match_datasets.sort_values(by='text_hash', ascending=False)
        assert len(match_datasets) >= 1
        most_common_dataset = match_datasets.index[0]
        most_common_dataset_count = match_datasets.iloc[0].text_hash
        most_common_friendly_name = FRIENDLY_NAME_MAP_SHORTER.get(most_common_dataset, most_common_dataset)
        table_rows.append([
            col.get(),
            int(df[col].sum()),
            match_rows['robot-truthful'].mean(),
            match_rows['robot-comfort'].mean(),
            match_rows['human-truthful'].mean(),
            match_rows['human-comfort'].mean(),
            f"{most_common_friendly_name} ({most_common_dataset_count})",
            ",".join(max_words),
        ])
    print(tabulate.tabulate(
        table_rows,
        headers=["Group", "Count", "RP", "RC", "HP", "HC"],
        floatfmt=".1f",
        tablefmt="latex",
    ))


def main():
    #df = get_filtered_joined_results()
    #print_free_resp()
    #dump_explans_to_csv()
    sort_filled_explans()
    make_table()


if __name__ == "__main__":
    main()