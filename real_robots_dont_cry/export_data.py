from pathlib import Path

from real_robots_dont_cry.classify_toy import get_classifiable_df, partition_df_hashes_split, \
    CLASS_SEED
from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, \
    filter_df_by_text_responses_threshold
from real_robots_dont_cry.join_gpt_challenge_results import get_joined_challenge_results
from real_robots_dont_cry.join_results import get_joined_results


def main():
    cur_file = Path(__file__).parent.absolute()
    out_root = cur_file / "data/dataexport"
    out_root.mkdir(exist_ok=True)
    print("out root", out_root)
    df = get_joined_results()
    df.to_csv(out_root / "joined_results_raw.csv", index=False)
    df = get_filtered_joined_results()
    df.to_csv(out_root / "joined_results_filt.csv", index=False)
    df = filter_df_by_text_responses_threshold(df, 3)
    df.to_csv(out_root / "joined_results_filt_threshed.csv", index=False)
    df = get_classifiable_df()
    df.to_csv(out_root / "classifiable_df_all.csv", index=False)
    train_df, val_df, test_df = partition_df_hashes_split(
        df,
        train_val_test_probs=(0.7, 0.15, 0.15),
        seed=CLASS_SEED,
    )
    train_df.to_csv(out_root / "classifiable_df_train.csv", index=False)
    val_df.to_csv(out_root / "classifiable_df_val.csv", index=False)
    test_df.to_csv(out_root / "classifiable_df_test.csv", index=False)
    ## LM
    df = get_joined_challenge_results()
    df.to_csv(out_root / "lm_eval_joined_results_raw.csv", index=False)
    df = get_filtered_joined_results()
    df = filter_df_by_text_responses_threshold(df, 3)
    df.to_csv(out_root / "lm_eval_joined_results_raw_filt_threshed.csv", index=False)


if __name__ == "__main__":
    main()