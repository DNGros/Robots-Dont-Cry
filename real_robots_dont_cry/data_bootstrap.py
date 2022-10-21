import math
import statistics

import more_itertools
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Callable, Union, Set, Iterable
from joblib import Memory
from collections import defaultdict
from prettyprinter import pprint

from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, matched_duplicate_rows, \
    filter_df_by_text_responses_threshold
import random
from dataclasses import dataclass

from real_robots_dont_cry.gensurvey import QuestionTopic
from real_robots_dont_cry.join_gpt_challenge_results import get_filtered_challenge_joined_results


@dataclass(frozen=True)
class BootstrappedSim:
    utterances: List['BootstrapedUtt']
    all_dataset_srcs: Set[str]
    all_bot_desc_cats: Set[str]


@dataclass(frozen=True)
class BootstrapedUtt:
    dataset_src: str
    bot_desc_cat: str
    is_fake_turn_a: bool
    q_kind_to_ans: Dict[str, List[int]]

    # @property
    # def robot_possible_ans(self) -> List[int]:
    #    return self.q_kind_to_ans['robot-truthful']

    # @property
    # def human_possible_ans(self) -> List[int]:
    #    return self.q_kind_to_ans['human-truthful']

    # @property
    # def robot_comfort_ans(self) -> List[int]:
    #    return self.q_kind_to_ans['robot-comfort']

    # @property
    # def human_comfort_ans(self) -> List[int]:
    #    return self.q_kind_to_ans['human-comfort']


class WeightedSampler:
    """Samples from a set of items by weight."""

    def __init__(self, items, weights):
        assert len(items) == len(weights)
        self._items = items
        total_weight = sum(weights)
        self._weights = [w / total_weight for w in weights]

    def sample(self):
        return random.choices(self._items, self._weights, k=1)[0]

    def __hash__(self):
        return hash((self._items, self._weights))

    def __eq__(self, other):
        return self._items == other._items and self._weights == other._weights

    def __str__(self):
        items_and_weights = sorted(list(
            zip(self._items, self._weights)),
            key=lambda x: x[1], reverse=True)
        items_and_weights = [
            (item, round(weight, 2))
            for item, weight in items_and_weights
        ]
        return f"WeightedSampler({items_and_weights})"

    def __repr__(self):
        return str(self)


def filter_sims(
    sims: List[BootstrappedSim],
    bot_desc_cat: str = None,
    dataset_src: str = None,
) -> List['BootstrappedSim']:
    return [
        BootstrappedSim(
            utterances=[
                utt for utt in sim.utterances
                if (
                    ((bot_desc_cat is None) or (utt.bot_desc_cat == bot_desc_cat))
                    and (dataset_src is None or (utt.dataset_src == dataset_src))
                )
            ],
            all_dataset_srcs=sim.all_dataset_srcs,
            all_bot_desc_cats=sim.all_bot_desc_cats
        )
        for sim in sims
    ]


def estimate_score_transition_probabilities(df):
    """Within an individual rater there is likely some variance in their ratings
    even for an identical utterance (if you ask them at different times they
    might give different answers).
    Likely depending on the score variance will
    be different. For example if someone was very confident and rated something
    a '5' then they will probably repeatedly rate it '5'. However, if they rated
    an utterance a '2', then they might be more likely to change their vote to a
    '1' or maybe a '3', but very unlikely to move their all the way up to a '5'.
    We will use the duplicate question to estimate this transition probability
    """
    if not len(df[df['is_duplicate']]) > 0:
        raise ValueError("Function should be supplied df with duplicates not prefiltered.")
    question_topic_to_score_to_new_score_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for orig, dup in matched_duplicate_rows(df):
        orig_ans = orig['ans']
        dup_ans = dup['ans']
        question_topic = orig['question_topic']
        question_topic_to_score_to_new_score_probs[question_topic][orig_ans][dup_ans] += 1
        question_topic_to_score_to_new_score_probs[question_topic][dup_ans][orig_ans] += 1
        # Extra bump to actual? Maybe should do a random walk instead?
        #question_topic_to_score_to_new_score_probs[question_topic][orig_ans][orig_ans] += 0.5
        #question_topic_to_score_to_new_score_probs[question_topic][dup_ans][dup_ans] += 0.5
    for question_topic, score_to_new_score_probs in question_topic_to_score_to_new_score_probs.items():
        for orig_ans, new_score_to_prob in score_to_new_score_probs.items():
            total = sum(new_score_to_prob.values())
            for new_score, prob in new_score_to_prob.items():
                new_score_to_prob[new_score] = prob / total
    # TODO: random walk to refine probabilities?
    return {
        question_topic: {
            orig_ans: WeightedSampler(list(new_score_to_prob.keys()), list(new_score_to_prob.values()))
            for orig_ans, new_score_to_prob in score_to_new_score_probs.items()
        }
        for question_topic, score_to_new_score_probs in
        question_topic_to_score_to_new_score_probs.items()
    }


def get_dataset_src_to_total_count(df):
    df = df[~df['is_human'] & (df['question_topic'] == "robot-truthful")]
    return df.groupby('dataset_src').text_hash.nunique().to_dict()


def make_bootstraped_samples(
    df,
    n_simulations: int,
    question_topic_to_score_to_new_score_sample: Dict[str, Dict[int, WeightedSampler]] = None,
) -> List[BootstrappedSim]:
    if question_topic_to_score_to_new_score_sample is None:
        question_topic_to_score_to_new_score_sample = estimate_score_transition_probabilities(df)

    all_embodiments = list(df['bot_desc_cat'].unique())
    out = []
    df_by_dataset_and_embodiment = {
        dataset_src: {
            embodiment: df[(df['dataset_src'] == dataset_src) & (df['bot_desc_cat'] == embodiment)]
            for embodiment in all_embodiments
        }
        for dataset_src in df['dataset_src'].unique()
    }
    dataset_src_to_total_count = get_dataset_src_to_total_count(df)
    print(dataset_src_to_total_count)
    for simulation in tqdm(range(n_simulations), desc="Simulations"):
        sim_utts = []
        for dataset, by_embodiment_df in df_by_dataset_and_embodiment.items():
            num_samples = dataset_src_to_total_count[dataset]
            for _ in range(num_samples):
                embodiment = random.choice(all_embodiments)
                avail_df = by_embodiment_df[embodiment]
                # Choose a utterance
                text_hash = random.choice(avail_df[~avail_df['is_human']]['text_hash'].unique())
                avail_df = avail_df[avail_df['text_hash'] == text_hash]
                # print(f"{text_hash=} {len(avail_df)=} {len(avail_df['task_id'].unique())=}")
                # Get simulated answers for that utterance
                q_kind_to_ans = defaultdict(list)
                num_resps = len(avail_df['task_id'].unique())
                for _ in range(num_resps):
                    # Choose a turker
                    task_id = random.choice(avail_df[~avail_df['is_human']]['task_id'].unique())
                    task_df = avail_df[avail_df['task_id'] == task_id]
                    # print(f"{task_id=} {len(avail_df)=}")
                    assert len(task_df) == 4  # one for each question kind
                    for row in task_df.itertuples():
                        assert not row.is_duplicate, "Duplicates should have been filtered out"
                        orig_to_sampler = question_topic_to_score_to_new_score_sample[row.question_topic]
                        if row.ans in orig_to_sampler:
                            sampler = orig_to_sampler[row.ans]
                            transitioned_ans = sampler.sample()
                        else:
                            transitioned_ans = row.ans
                        #print(f"{row.question_topic=} {row.ans=} {transitioned_ans=}")
                        q_kind_to_ans[row.question_topic].append(transitioned_ans)
                sim_utts.append(BootstrapedUtt(
                    dataset_src=dataset,
                    bot_desc_cat=embodiment,
                    is_fake_turn_a=avail_df.iloc[0]['is_fake_turn_a'],
                    q_kind_to_ans=dict(q_kind_to_ans),
                ))
                # Choose votes
                #   apply single annotator transition model
        out.append(BootstrappedSim(
            sim_utts,
            all_dataset_srcs=set(df['dataset_src'].unique()),
            all_bot_desc_cats=set(df['bot_desc_cat'].unique()),
        ))
    return out


# def _find_samples_per_dataset(df):
#    df = df[df['q_kind'] == df.iloc[0]['q_kind']]
#    samples_per_dataset = df.groupby('dataset_src').count()['question_id']
#    #assert samples_per_dataset.max() == samples_per_dataset.min()
#    return samples_per_dataset.min()


def bootstrapped_majority_prob(
    responses: List[int],
    transition_probs: Dict[int, WeightedSampler],
    samples: int = 200
) -> float:
    return statistics.mean(
        1 if determine_majority_possible_judgement(sample) else 0
        for sample in iterate_response_list_bootstrap_samples(
            responses, transition_probs, samples
        )
    )


def iterate_response_list_bootstrap_samples(
    responses: List[int],
    transition_probs: Dict[int, WeightedSampler],
    samples: int
) -> Iterable[Iterable[int]]:
    yield from more_itertools.windowed(
        (
            transition_probs[v].sample() if transition_probs is not None else v
            for v in random.choices(responses, k=len(responses) * samples)
        ),
        n=len(responses),
        step=len(responses),
    )


def determine_majority_possible_judgement(responses: List[int]) -> bool:
    return (
        #(len([r for r in responses if r >= 4]) >= len(responses) / 2)
        (statistics.median(responses) >= 3)
        and (statistics.mean(map(float, responses)) >= 3)  # means something like [5, 3, 3, 1, 1] not True
    )


def determine_vals_high(responses: List[int]) -> bool:
    return statistics.mean(responses) >= 4


@dataclass(frozen=True)
class ConfIntervalDatum:
    median: float
    c_low: float
    c_high: float
    interval_size: float

    def format(self) -> str:
        return f"{self.median:.03f} " \
               f"(C{self.interval_size} {self.c_low:0.02f}-{self.c_high:0.02f})"

def calc_median_and_confidence_interval(
    sims: List[BootstrappedSim],
    sim_to_val: Callable[[BootstrappedSim], Union[float, bool]],
    confidence_interval_size: float = 90,
) -> ConfIntervalDatum:
    vals = np.array(list(map(sim_to_val, sims)))
    assert 0 < confidence_interval_size <= 100
    return ConfIntervalDatum(
        median=np.percentile(vals, 50),
        c_low=np.percentile(vals, 50 - confidence_interval_size / 2),
        c_high=np.percentile(vals, 50 + confidence_interval_size / 2),
        interval_size=confidence_interval_size,
    )


cachedir = 'bootstrap_cache'
diskcache = Memory(cachedir, verbose=2)


@diskcache.cache
def get_data_bootstrap_sims(
    n_simulations: int = 1000,
    df_prefilter_func: Callable = None,
    challenge: bool = False
) -> List[BootstrappedSim]:
    if not challenge:
        df = get_filtered_joined_results(remove_duplicates=False)
    else:
        print("challenge")
        df = get_filtered_challenge_joined_results(remove_duplicates=False)
    if df_prefilter_func is not None:
        df = df_prefilter_func(df)
    question_topic_to_score_to_new_score_sample = estimate_score_transition_probabilities(df)
    pprint(question_topic_to_score_to_new_score_sample)
    df = df[~df['is_duplicate']]
    df = filter_df_by_text_responses_threshold(df, threshold_responses=3)
    sims = make_bootstraped_samples(
        df,
        n_simulations,
        #samples_per_dataset=12,
        #samples_per_ex=5,
        #samples_per_dataset=100,
        question_topic_to_score_to_new_score_sample=question_topic_to_score_to_new_score_sample
    )
    return sims


def frac_ans_majority(
    utts: List[BootstrapedUtt],
    q_kind: QuestionTopic
) -> float:
    return statistics.mean(
        1 if determine_majority_possible_judgement(utt.q_kind_to_ans[q_kind.value]) else 0
        for utt in utts
    )


def question_mean(
    utts: List[BootstrapedUtt],
    q_kind: QuestionTopic
) -> float:
    return statistics.mean(
        statistics.mean(map(float, utt.q_kind_to_ans[q_kind.value]))
        for utt in utts
    )


def frac_ans_high(
    utts: List[BootstrapedUtt],
    q_kind: QuestionTopic
) -> float:
    utts_high = [
        determine_vals_high(utt.q_kind_to_ans[q_kind.value])
        for utt in utts
    ]
    return sum(utts_high) / len(utts_high)


def main():
    sims = get_data_bootstrap_sims(n_simulations=51)
    #print(calc_median_and_confidence_interval(
    #    sims,
    #    lambda sim: frac_ans_majority(sim.utterances, QuestionTopic.ROBOT_POSSIBLE),
    #))
    print(calc_median_and_confidence_interval(
        sims,
        lambda sim: frac_ans_high(sim.utterances, QuestionTopic.ROBOT_COMFORTABLE),
    ))


def bootstrap_hypothesis_test_tailed(
    sims: List[BootstrappedSim],
    lt_value_func: Callable[[BootstrappedSim], float],
    gt_value_func: Callable[[BootstrappedSim], float],
):
    lt_vals = np.array([lt_value_func(s) for s in sims])
    gt_vals = np.array([gt_value_func(s) for s in sims])
    return (lt_vals > gt_vals).mean()


def sim_mean_filtered(
    utt_filter_include: Callable[[BootstrapedUtt], bool],
    mean_extractor: Callable[[List[BootstrapedUtt]], float],
):
    def func(sim):
        utts = [utt for utt in sim.utterances if utt_filter_include(utt)]
        return mean_extractor(utts)
    return func


def utt_has_emboddiment(bot_desc_cat: str) -> Callable[[BootstrapedUtt], bool]:
    return lambda utt: utt.bot_desc_cat == bot_desc_cat


def find_metric_results_for_sim(filtered_sims: List[BootstrappedSim]) -> Dict[str, ConfIntervalDatum]:
    calc = lambda func: calc_median_and_confidence_interval(filtered_sims, func)
    return {
        'Robot Possible Majority': calc(
            lambda sim: frac_ans_majority(
                sim.utterances, QuestionTopic.ROBOT_POSSIBLE),
        ),
        'Robot Possible Mean': calc(
            lambda sim: question_mean(
                sim.utterances, QuestionTopic.ROBOT_POSSIBLE),
        ),
        'Human Possible Majority': calc(
            lambda sim: frac_ans_majority(
                sim.utterances, QuestionTopic.HUMAN_POSSIBLE),
        ),
        'Human Possible Mean': calc(
            lambda sim: question_mean(
                sim.utterances, QuestionTopic.HUMAN_POSSIBLE),
        ),
        'Robot High Comfortable': calc(
            lambda sim: frac_ans_high(
                sim.utterances, QuestionTopic.ROBOT_COMFORTABLE),
        ),
        'Human High Comfortable': calc(
            lambda sim: frac_ans_high(
                sim.utterances, QuestionTopic.HUMAN_COMFORTABLE),
        ),
        'Robot Comfortable Majority': calc(
            lambda sim: frac_ans_majority(
                sim.utterances, QuestionTopic.ROBOT_COMFORTABLE),
        ),
        'Robot Comfortable Mean': calc(
            lambda sim: question_mean(
                sim.utterances, QuestionTopic.ROBOT_COMFORTABLE),
        ),
        'Human Comfortable Majority': calc(
            lambda sim: frac_ans_majority(
                sim.utterances, QuestionTopic.HUMAN_COMFORTABLE),
        ),
        'Human Comfortable Mean': calc(
            lambda sim: question_mean(
                sim.utterances, QuestionTopic.HUMAN_COMFORTABLE),
        ),
    }


if __name__ == "__main__":
    main()