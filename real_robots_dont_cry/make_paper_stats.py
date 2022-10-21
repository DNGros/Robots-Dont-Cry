from pathlib import Path
from typing import Dict
import pandas as pd

from real_robots_dont_cry.classify_toy import get_classifiable_df
from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, \
    filter_df_by_text_responses_threshold
from real_robots_dont_cry.gensurvey import QuestionTopic
from real_robots_dont_cry.join_results import get_joined_results
from real_robots_dont_cry.new_json_filter_by_qual_check import get_passing_non_passing_df

cur_file = Path(__file__).parent.absolute()


def vars_to_latex(vars_dict: Dict) -> str:
    """
    Create latex commands for each variable.
    """
    latex_str = ""
    for key, value in vars_dict.items():
        latex_str += f"\\newcommand{{\\{key}}}{{{value}\\xspace}}\n"
    return latex_str


def add_qual_stats(stats):
    df, non_passing_df, passing_df = get_passing_non_passing_df(
        consider_free_resp=True,
    )
    df = passing_df
    has_explan = df[df.user_explanation.notnull()]
    stats['totalExplanCount'] = len(has_explan)
    df = pd.read_csv(cur_file / "generations/user_explans_v2_fill.csv")
    stats['extractedExplanCount'] = len(df)
    stats['explanHasTwoCount'] = len(df[df['Has Label'] >= 2])


def add_general_stats(stats):
    df = get_joined_results()
    df = df[df['worker_id_hash'].notnull()]
    stats['numSurveyResponses'] = len(df.task_id.unique())
    max_worker_hit_num = 3
    df = df[df['worker_hit_num'] <= max_worker_hit_num]
    num_after_worker_filter = len(df.task_id.unique())
    stats['numMultiWorker'] = (stats['numSurveyResponses'] - num_after_worker_filter)
    df = get_filtered_joined_results()
    stats['numSurveysFiltered'] = len(df.task_id.unique())
    stats['filterSuccessRate'] = str(round(
        stats['numSurveysFiltered'] / num_after_worker_filter * 100,
        0
    )) + "\%"
    stats['numIndividuals'] = len(df.worker_id_hash.unique())
    df = filter_df_by_text_responses_threshold(df, threshold_responses=3)
    stats['numDialogs'] = len(df.text_hash.unique())
    stats['numDialogsPerDatasourceMin'] = df.groupby(['dataset_src']).text_hash.nunique().min()
    stats['numDialogsPerDatasourceMax'] = df.groupby(['dataset_src']).text_hash.nunique().max()
    stats['averageNumRespsPerQuestion'] = round(
        df.groupby(['text_hash', 'bot_desc_cat', 'question_topic']).count().turn_a.mean(),
        1
    )
    stats['totalLikertRatings'] = len(df)


def num_impossible_all():
    df = get_filtered_joined_results()
    df = filter_df_by_text_responses_threshold(df, 3)
    class_df = get_classifiable_df()
    class_df = class_df[class_df.question_topic == QuestionTopic.ROBOT_POSSIBLE.value]
    print(len(class_df), "all len")
    print(class_df.text_hash.nunique(), "text hash")
    print(class_df[class_df['majority_prob'] < 0.5].text_hash.nunique(), "impossible")


def main():
    stats = {}
    #num_impossible_all()
    #exit()
    add_general_stats(stats)
    add_qual_stats(stats)
    ###
    print("-" * 80)
    print(vars_to_latex(stats))


if __name__ == "__main__":
    main()