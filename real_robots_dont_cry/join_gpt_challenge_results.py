import pandas as pd
import json
import pandas as pd
import typing
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, fields, is_dataclass
from prettyprinter import pprint
import prettyprinter
from collections import Counter

from real_robots_dont_cry.explore_quality_check import filter_df_by_pass_quality, \
    is_a_quality_check_row, filter_df_by_text_responses_threshold
from real_robots_dont_cry.explore_worker_ids import hash_worker_id
from real_robots_dont_cry.join_results import get_used_surveys, resp_to_df
from real_robots_dont_cry.worker_id_load import get_transactions_df
from util.sampling import deterministic_hash

verbose = False

cur_file = Path(__file__).parent.absolute()


def get_challenge_raw_results():
    return [
        *json.loads((cur_file / "responses/rdc_results_lmphase1.json").read_text()),
        *json.loads((cur_file / "responses/rdc_results_lmphase2.json").read_text()),
    ]


def get_joined_challenge_results() -> pd.DataFrame:
    responses = get_challenge_raw_results()
    if verbose:
        print(f"{len(responses)=}")
        print(f"{type(responses)=}")
    surveys = [
        *get_used_surveys("generations/robotcry-lm-gens-v1.json"),
        *get_used_surveys("generations/challenge_after_phase_1.json"),
    ]
    return resp_to_df(responses, surveys)


def get_filtered_challenge_joined_results(
    remove_duplicates: bool = True,
    max_worker_hit_num: int = 3,
    max_worker_text_num: int = 1,
    filter_by_free_resp_explan_long_enough: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    df = get_joined_challenge_results()
    df = filter_df_by_pass_quality(
        df,
        max_worker_hit_num=max_worker_hit_num,
        max_worker_text_num=max_worker_text_num,
        consider_free_resp=filter_by_free_resp_explan_long_enough,
        verbose=verbose,
    )
    df = df[~is_a_quality_check_row(df)]
    if remove_duplicates:
        df = df[~df['is_duplicate']]
    return df


def main():
    results = get_filtered_challenge_joined_results()
    results = filter_df_by_text_responses_threshold(results, 3)
    print(results.task_id.nunique(), "tasks")
    print(results.worker_id_hash.nunique(), "workers")
    print(len(results), "likert")


if __name__ == "__main__":
    main()
