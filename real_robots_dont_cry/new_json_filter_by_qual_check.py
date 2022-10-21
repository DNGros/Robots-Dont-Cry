import json
import random
from pathlib import Path
from typing import List

from real_robots_dont_cry.explore_quality_check import filter_df_by_pass_quality, \
    get_task_ids_that_pass_all_checks
from real_robots_dont_cry.gensurvey import surveys_as_json, SurveyMetad, LAST_PAGE_TEXT
from real_robots_dont_cry.join_gpt_challenge_results import get_filtered_challenge_joined_results, \
    get_joined_challenge_results
from real_robots_dont_cry.join_results import get_joined_results, get_f1_raw_responses, get_used_surveys
from pprint import pprint
import dataclasses

from real_robots_dont_cry.resp_count import sorted_surveys_by_need

cur_file = Path(__file__).parent.absolute()


def replace_last_page_text(surveys: List[SurveyMetad], new_last_page_text):
    out = []
    for survey in surveys:
        out.append(dataclasses.replace(survey, last_page_text=new_last_page_text))
    return out


def get_passing_non_passing_df(
    consider_free_resp=True,
    challenge: bool = False,
):
    if not challenge:
        df = get_joined_results()
    else:
        df = get_joined_challenge_results()
    pass_task_ids = get_task_ids_that_pass_all_checks(
        df,
        consider_free_resp=consider_free_resp,
    )
    non_passing_df = df[~df.task_id.isin(pass_task_ids)]
    passing_df = df[df.task_id.isin(pass_task_ids)]
    return df, non_passing_df, passing_df


def get_raw_results_no_passing():
    #surveys = get_used_surveys("generations/robotcry-survey-full-v15.json")
    surveys = get_used_surveys("generations/robotcry-lm-gens-v2.json")
    challenge = True
    df, non_passing_df, passing_df = get_passing_non_passing_df(
        consider_free_resp=True,
        challenge=challenge
    )
    surveys = [
        survey for survey in surveys
        if (
            survey.id_hash not in passing_df.survey_id_hash.unique()
        )
    ]
    surveys = replace_last_page_text(surveys, LAST_PAGE_TEXT)
    surveys = sorted_surveys_by_need(surveys, challenge=challenge)
    print(len(surveys))
    surveys = surveys[:130]
    print("Sorted!")
    random.shuffle(surveys)
    print(f"{len(surveys)=}")
    (cur_file / "generations/challenge_after_phase_1.json").write_text(
        surveys_as_json(surveys)
    )


def main():
    get_raw_results_no_passing()


if __name__ == "__main__":
    main()
