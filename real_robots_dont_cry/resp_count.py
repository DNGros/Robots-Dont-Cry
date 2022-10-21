from typing import List

from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, \
    filter_df_by_text_responses_threshold, is_a_quality_check_row
from collections import Counter, defaultdict

from real_robots_dont_cry.gensurvey import SurveyMetad, DialoguePageMetad
from real_robots_dont_cry.join_gpt_challenge_results import get_filtered_challenge_joined_results
from real_robots_dont_cry.join_results import get_f1_raw_responses, get_used_surveys
from real_robots_dont_cry.visresults import print_all


def sorted_surveys_by_need(
    surveys: List[SurveyMetad],
    challenge: bool = False
) -> List[SurveyMetad]:
    if not challenge:
        df = get_filtered_joined_results()
    else:
        df = get_filtered_challenge_joined_results()
    df = df[(~df['is_human']) & (df['question_topic'] == "robot-truthful") & (~is_a_quality_check_row(df))]
    g_by_bot_desc_and_hash = df.groupby(['bot_desc_cat', 'text_hash'])
    # Take the count of the answers for each text_hash and put it as a new row
    #  in the dataframe
    text_hash_count_to_score = defaultdict(lambda: 0.0, {
        0: 1.00,
        1: 0.5,
        2: .35,
        3: 0.02,
        4: 0,
    })
    #df['text_ans_count_score'] = df['text_ans_count'].map(scoring)
    #import pandasgui
    #g = df.groupby('survey_id_hash').text_ans_count_score.sum()
    #pandasgui.show(g)
    #print(df.head())

    def get_survey_need(survey: SurveyMetad) -> float:
        need_scores = []
        vals = []
        for page in survey.pages:
            if not isinstance(page, DialoguePageMetad):
                continue
            if page.turn_metad.turn.dataset_src == "quality_check":
                continue
            text_hash = page.turn_metad.turn.calc_text_hash()
            bot_desc_cat = page.questions[0].bot_desc_cat
            try:
                group = g_by_bot_desc_and_hash.get_group((bot_desc_cat, text_hash))
                val = len(group)
            except KeyError:
                val = 0
            #print(val)
            vals.append(val)
            #if val == 0:
            #    print(page)
            need_scores.append(text_hash_count_to_score[val])
        need_score = sum(need_scores)
        if need_score > 2.7:
            print(vals)
            print("need score:", need_score)
        return need_score

    score_v = zip(map(get_survey_need, surveys), surveys)
    score_v = sorted(score_v, key=lambda v: v[0], reverse=True)
    print([(score, survey.id_hash) for score, survey in score_v[:50]])
    return [v[1] for v in score_v]


def print_survey_resp_count():
    #df = get_filtered_joined_results()
    df = get_filtered_challenge_joined_results()
    print("df len", len(df))
    df = filter_df_by_text_responses_threshold(df, 3)
    print("HITS len", len(df.task_id.unique()))
    print("Workers len", len(df.worker_id_hash.unique()))
    df = df[~df['is_human'] & (df['question_topic'] == "robot-truthful")]
    g = df.groupby(['text_hash', 'resp_cat']).ans.count()
    print(g.sort_values(ascending=False))
    ans_count = Counter(g)
    print_ans_counts_table(ans_count)
    print(f"{df['text_hash'].nunique()=}")
    print(df.groupby('worker_id').count())
    print(df[
              (df['resp_cat'] == "truthful_r-chatbot")
              | (df['resp_cat'] == "truthful_r-humanoid")
          ].groupby('dataset_src').text_hash.nunique())
    print(f"{df.worker_id.nunique()} workers")
    print(f"{df.text_hash.nunique()} text_hash")
    print(f"{len(df)} rows")


def print_ans_counts_table(ans_count: Counter):
    print("Num Responses")
    print("Num_resps\tUtt_count\tCumulative")
    cumulative = 0
    for num_resps in sorted(ans_count.keys(), reverse=True):
        count = ans_count[num_resps]
        cumulative += count
        print(f"{num_resps}:\t{count}\t{cumulative}")


def main():
    #df = get_filtered_joined_results()
    #df = filter_df_by_text_responses_threshold(df, 3)
    #print(df.q_kind.unique())
    #print(sum(
    #    (df[df['q_kind'] == 'truthful'].groupby('text_hash').ans.median() < 3)
    #    | (df[df['q_kind'] == 'truthful'].groupby('text_hash').ans.mean() < 3)
    #))
    print(print_survey_resp_count())
    surveys = get_used_surveys("generations/robotcry-survey-full-v15.json")
    surveys = sorted_surveys_by_need(surveys)


if __name__ == '__main__':
    main()