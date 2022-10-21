from prettyprinter import pprint

from real_robots_dont_cry.data_bootstrap import estimate_score_transition_probabilities, \
    WeightedSampler
from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results


def main():
    df = get_filtered_joined_results(remove_duplicates=False)
    question_topic_to_score_to_new_score_sample = estimate_score_transition_probabilities(df)
    pprint(question_topic_to_score_to_new_score_sample)
    for topic, start_map in question_topic_to_score_to_new_score_sample.items():
        rows_cols = []
        for score in range(1, 6):
            rows_cols.append([])
            sampler: WeightedSampler = start_map[score]
            d = dict(zip(sampler._items, sampler._weights))
            for trans in range(1, 6):
                rows_cols[-1].append(d.get(trans, 0))
        print(topic)
        print("\n".join(",".join(map(str, r)) for r in rows_cols))


if __name__ == "__main__":
    main()