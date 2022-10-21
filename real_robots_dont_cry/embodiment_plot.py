import statistics
from typing import List, Callable
from collections import defaultdict, Counter
from pprint import pprint

from real_robots_dont_cry.data_bootstrap import get_data_bootstrap_sims, filter_sims, \
    find_metric_results_for_sim, iterate_response_list_bootstrap_samples, \
    estimate_score_transition_probabilities, bootstrap_hypothesis_test_tailed, sim_mean_filtered, \
    utt_has_emboddiment, question_mean
from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, \
    filter_df_by_text_responses_threshold
from real_robots_dont_cry.gensurvey import QuestionTopic


def _prefilt_func(df):
    return df[~df['dataset_src'].isin(('multi_woz_v22',))]


def _prefilt_func_only_bottom(df):
    return df[df['dataset_src'].isin(('msc', "reddit_small", "personachat_personas"))]


def make_diff_cat_table_embodiment():
    df = get_filtered_joined_results(remove_duplicates=False)
    transition_probs = estimate_score_transition_probabilities(df)
    df = df[~df['is_duplicate']]
    threshold_responses = 3
    q_topic_to_cat_to_count = defaultdict(lambda: Counter())
    for q_topic in (QuestionTopic.ROBOT_POSSIBLE, QuestionTopic.ROBOT_COMFORTABLE):
        fdf = df[df['question_topic'] == q_topic.value]
        cat_to_count = q_topic_to_cat_to_count[q_topic]
        for text_hash, text_group in fdf.groupby('text_hash'):
            humanoid_ans = list(text_group[text_group['bot_desc_cat'] == "humanoid"].ans)
            chatbot_ans = list(text_group[text_group['bot_desc_cat'] == "chatbot"].ans)
            if len(humanoid_ans) < threshold_responses or len(chatbot_ans) < threshold_responses:
                continue
            cat = get_bootstrap_aware_cat(
                humanoid_ans, chatbot_ans,
                transition_probs_a=transition_probs[q_topic.value],
                transition_probs_b=transition_probs[q_topic.value],
                #cat_confidence_threshold=0.75,
                cat_confidence_threshold=0.9,
                samples=100,
            )
            if cat != None:
                print("Question topic:", q_topic)
                print("CAT", cat, "humanoid", humanoid_ans, "chatbot", chatbot_ans)
                print("turn_a:", text_group['turn_a'].iloc[0])
                print("turn_b:", text_group['turn_b'].iloc[0])
            cat_to_count[cat] += 1

    pprint(q_topic_to_cat_to_count)


def get_bootstrap_aware_cat(
    ans_a,
    ans_b,
    transition_probs_a,
    transition_probs_b,
    cat_confidence_threshold: float,
    samples: int = 100,
):
    base_cats = bootstrap_sample_cmp_ans(
        ans_a, ans_b, transition_probs_a, transition_probs_b, samples)
    #print("base cats", base_cats)
    assert sum(base_cats.values()) == samples
    most_cat, most_count = base_cats.most_common(n=1)[0]
    #print(most_cat, most_count)
    if most_count / samples < cat_confidence_threshold:
        return None
    return most_cat


def bootstrap_sample_cmp_ans(
    ans_a,
    ans_b,
    transition_probs_a,
    transition_probs_b,
    samples: int = 100
):
    cats = Counter()
    for a_sample, b_sample in zip(
        iterate_response_list_bootstrap_samples(ans_a, transition_probs_a, samples),
        iterate_response_list_bootstrap_samples(ans_b, transition_probs_b, samples),
        #iterate_response_list_bootstrap_samples(ans_a, None, samples),
        #iterate_response_list_bootstrap_samples(ans_b, None, samples),
    ):
        cats[_cmp_ans_lists(a_sample, b_sample)] += 1
    return cats


def _cmp_ans_lists(
    ans_a: List[int],
    ans_b: List[int],
    agg_func: Callable = statistics.median,
    thresh: float = 1.0,
) -> int:
    """
    returns -1 if ans_a is greater than ans_b by threshold.
    Returns 0 if within same threshold
    Returns 1 if ans_b is less than ans_a by threshold

    Uses a configurable agg function to aggregate the list
    """
    agg_a = agg_func(ans_a)
    agg_b = agg_func(ans_b)
    diff = agg_a - agg_b
    if diff >= thresh:
        return -1
    elif thresh > diff > -thresh:
        return 0
    else:
        assert diff <= thresh, diff
        return 1


#def filt_df_has_multiple_embodiments(df):
#    return df[
#        df.groupby(['text_hash'])['bot_desc_cat'].transform('nunique') > 1
#    ]


def main():
    #make_diff_cat_table_embodiment()
    #exit()
    #df = get_filtered_joined_results()
    #print(df.groupby(["question_topic", "bot_desc_cat"]).ans.mean())
    sims = get_data_bootstrap_sims(
        n_simulations=1000,
        #df_prefilter_func=_prefilt_func
        df_prefilter_func=_prefilt_func_only_bottom
    )
    print("SDFSDFSDF")
    print(sims[0].all_bot_desc_cats)
    p_val = bootstrap_hypothesis_test_tailed(
        sims,
        lt_value_func=sim_mean_filtered(
            utt_has_emboddiment("chatbot"),
            lambda utts: question_mean(utts, QuestionTopic.ROBOT_POSSIBLE),
        ),
        gt_value_func=sim_mean_filtered(
            utt_has_emboddiment("humanoid"),
            lambda utts: question_mean(utts, QuestionTopic.ROBOT_POSSIBLE),
        ),
    )
    print("P-value", p_val)
    exit()
    #for dataset in sims[0].all_dataset_srcs:

    #    print("Dataset", dataset)
    #    for bot in sims[0].all_bot_desc_cats:
    #        print(f"Bot", bot)
    #        filt_sims = filter_sims(sims, bot_desc_cat=bot, dataset_src=dataset)
    #        metrics = find_metric_results_for_sim(filt_sims)
    #        for metric, interval in metrics.items():
    #            if "Human" in metric:
    #                continue
    #            if "High" in metric:
    #                continue
    #            print(f"{metric}: ", interval.format())

    for bot in sims[0].all_bot_desc_cats:
        print(f"Bot", bot)
        filt_sims = filter_sims(sims, bot_desc_cat=bot)
        metrics = find_metric_results_for_sim(filt_sims)
        for metric, interval in metrics.items():
            if "Human" in metric:
                continue
            if "High" in metric:
                continue
            if "Mean" not in metric:
                continue
            print(f"{metric}: ", interval.format())


if __name__ == "__main__":
    main()