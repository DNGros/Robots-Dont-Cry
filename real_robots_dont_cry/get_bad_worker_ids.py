from pathlib import Path
import json
from real_robots_dont_cry.explore_quality_check import filter_df_by_pass_quality
from real_robots_dont_cry.join_results import get_joined_results
from real_robots_dont_cry.new_json_filter_by_qual_check import get_passing_non_passing_df

cur_file = Path(__file__).parent.absolute()


def main():
    df, non_passing_df, passing_df = get_passing_non_passing_df(
        consider_free_resp=True,
    )
    bad_ids = [
        wid
        for wid in set(non_passing_df['worker_id'].unique())
        if wid is not None
    ]
    print(bad_ids)
    (cur_file / "generations/after_phase_15_bad_ids_frsp.json").write_text(
        json.dumps(bad_ids)
    )


if __name__ == "__main__":
    main()

