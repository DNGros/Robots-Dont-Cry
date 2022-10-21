from collections import Counter
from pprint import pprint

import pandas as pd

#from real_robots_dont_cry.classify_toy import partition_df_hashes_split, CLASS_SEED
from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, \
    filter_df_by_text_responses_threshold
from real_robots_dont_cry.visresults import print_all


#def select_challenge_datset():
#    df = get_filtered_joined_results()
#    df = filter_df_by_text_responses_threshold(df, threshold_responses=3)
#    train_df, val_df, test_df = partition_df_hashes_split(
#        df, (.7, .15, .15), seed=CLASS_SEED,
#    )
#    #sort_df = pd.concat([val_df, test_df])
#    #sort_df = test_df
#    sort_df = test_df
#    sort_df = sort_df[~sort_df['is_human']]
#    sort_df = sort_df[~sort_df['is_fake_turn_a']]
#    sort_df = sort_df[sort_df.turn_a.str.count(' ') >= 5]
#    print("LOW EXAMPLES")
#    lowest = sort_df.groupby('text_hash').ans.mean().sort_values(ascending=True)
#    lowest = lowest.iloc[:3]
#    print("Num lowest: {}".format(len(lowest)))
#    dataset_counter = Counter()
#    question_mark_count = 0
#    for i in range(len(lowest)):
#        print("Lowest #{}".format(i))
#        print_all(df[df['text_hash'] == lowest.index[i]])
#        dataset_counter[df[df['text_hash'] == lowest.index[i]].iloc[0].dataset_src] += 1
#        has_question_mark = df[df['text_hash'] == lowest.index[i]].iloc[0].turn_a.count('?') > 0
#        if has_question_mark:
#            question_mark_count += 1
#    print("Dataset counter:")
#    pprint(dataset_counter)
#    print("Question mark count", question_mark_count)
#    # return the sort_df if in the lowest
#    return sort_df[sort_df['text_hash'].isin(lowest.index)]


def main():
    #challenge_df = select_challenge_datset()
    #print("Challenge df:")
    #print(challenge_df)
    #exit()
    df = get_filtered_joined_results()
    df = filter_df_by_text_responses_threshold(df, threshold_responses=3)
    #df = df[df['dataset_src'] == 'wizard_of_wikipedia']
    sort_df = df
    sort_df = sort_df[~df['is_human']]
    sort_df = sort_df[~df['is_fake_turn_a']]
    print("HIGH EXAMPLES")
    highest = sort_df.groupby('text_hash').ans.mean().sort_values(ascending=False)
    for i in range(5):
        print_all(df[df['text_hash'] == highest.index[i]])
    print("#" * 80)
    print("LOW EXAMPLES")
    lowest = sort_df.groupby('text_hash').ans.mean().sort_values(ascending=True)
    for i in range(int(len(lowest) / 5)):
        print("Lowest #{}".format(i))
        print_all(df[df['text_hash'] == lowest.index[i]])


if __name__ == "__main__":
    main()
