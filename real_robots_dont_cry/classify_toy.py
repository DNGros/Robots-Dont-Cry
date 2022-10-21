import statistics
import random
from collections import defaultdict
import inspect
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import transformers
from joblib import Memory
from tqdm import tqdm
from transformers import RobertaTokenizer, AutoTokenizer

import classify_text_plz.quickclassify
from baselines.runbaseline import GramModelSuposedTrained
from classify_text_plz.classifiers.autoregressive_prompt_model import PlzPromptModelMaker
from classify_text_plz.classifiers.bow_logreg import BowModelLogRegressModel
from classify_text_plz.classifiers.deeplearn.bertbaseline import BertlikeTrainedModel, BertlikeModelMaker
from classify_text_plz.classifiers.deeplearn.lmpredictor import get_goose_lm, get_open_ai_lm
from classify_text_plz.classifiers.stupid_classifiers import AlwaysMostCommonModelTrained, RandomGuessTrained, \
    MostCommonClassModelMaker, RandomGuessModelMaker
from classify_text_plz.classifiers.tfidf_nn import TfidfKnnModelMaker
from classify_text_plz.dataing import MyTextData, create_my_data_splits, MyTextDataSplit, DataSplit, \
    MyTextDataSplitSoftBinary
from classify_text_plz.evaling import PlzEvaluator, Accuracy, Recall, Precision, EvalResult, BrierScore, \
    SoftAwareAccuracy, SoftAwareRecall, SoftAwarePrecision, get_soft_roc_curve, custom_plot_roc_curve, \
    SoftAwareRocAuc
from classify_text_plz.modeling import Prediction
from real_robots_dont_cry.data_bootstrap import determine_majority_possible_judgement, \
    estimate_score_transition_probabilities, bootstrapped_majority_prob, \
    iterate_response_list_bootstrap_samples
from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results, \
    filter_df_by_text_responses_threshold
import seaborn as sns
import numpy as np

from real_robots_dont_cry.gensurvey import QuestionTopic, QuestionText
import dill as pickle

cachedir = 'classify_data_cache'
diskcache = Memory(cachedir, verbose=2)


CLASS_SEED = 1234


def values_to_classifiable_str(turn_a, turn_b, question_text, bot_desc_cat):
    if isinstance(question_text, QuestionText):
        question_text = question_text.value
    segmented_text = (
        f"Human: {turn_a}",
        f"{bot_desc_cat}: {turn_b}",
        f"\n"
        f"Question: {question_text.replace(' R ', ' the ' + bot_desc_cat + ' ').lower()}",
    )
    return "\n".join(segmented_text)


@diskcache.cache
def get_classifiable_df(
    samples=10_000,
    # The result is tied to the values_to_classifiable func, but changes
    #    to this are not by default noticed by the diskcache.
    segment_func_text_for_cache=str(inspect.getsource(values_to_classifiable_str)),
):
    f = len(segment_func_text_for_cache)
    df = get_filtered_joined_results(remove_duplicates=False)
    question_topic_to_score_to_new_score_sample = estimate_score_transition_probabilities(df)
    pprint(question_topic_to_score_to_new_score_sample)
    df = df[~df['is_duplicate']]
    df = filter_df_by_text_responses_threshold(df, 3)
    df = df[~df['is_human']]
    g = df.groupby(['text_hash', 'question_topic', 'bot_desc_cat'])
    class_vals = []
    for (text_hash, question_topic, bot_desc_cat), group in tqdm(g):
        # print(text_hash, group)
        ans = list(group.ans)
        # ans = ans * 2
        class_vals.append({
            'text_hash': text_hash,
            'resp_cat': group['resp_cat'].iloc[0],
            'question_topic': question_topic,
            'bot_desc_cat': bot_desc_cat,
            'turn_a': group['turn_a'].iloc[0],
            'turn_b': group['turn_b'].iloc[0],
            'question_text': group['question_text'].iloc[0],
            'answers': ans,
            'ans_mean': statistics.mean(ans),
            'base_majority_vote': determine_majority_possible_judgement(ans),
            'majority_prob': bootstrapped_majority_prob(
                ans,
                question_topic_to_score_to_new_score_sample[group['question_topic'].iloc[0]],
                samples=samples,
            ),
            'joined_text': values_to_classifiable_str(
                group['turn_a'].iloc[0],
                group['turn_b'].iloc[0],
                group['question_text'].iloc[0],
                bot_desc_cat,
            )
        })
        class_vals[-1]['majority_prob_descrete'] = class_vals[-1]['majority_prob'] > 0.5
    return pd.DataFrame(class_vals)


def explore_majority_prob(class_df):
    # Print number of each base_majority_vote
    print(class_df.groupby('base_majority_vote').count())
    print(class_df.groupby('majority_prob_descrete').count())
    print(class_df.groupby(['question_topic', 'bot_desc_cat', 'base_majority_vote']).count())
    plt.figure(figsize=(4, 2))
    sns.histplot(class_df['majority_prob'], bins=10)
    # label x axis every 0.1 from 0 to 1
    plt.xticks(np.arange(0, 1.1, 0.1))
    # set the x axis
    plt.xlabel("Utterance's Prob of Majority Possible")
    plt.ylabel("#Utterances")
    plt.tight_layout()
    plt.show()
    # class_df = class_df[class_df['majority_prob_descrete'] != class_df['base_majority_vote']]
    #for r in class_df.sample(10).to_dict(orient='records'):
    #    print("Human:", r['turn_a'])
    #    print(f"{r['bot_desc_cat']}:", r['turn_b'])
    #    print(r['resp_cat'])
    #    print(r['answers'])
    #    print('Prob:', r['majority_prob'])
    #    print("-" * 20)


def do_rdc_from_plz_data(plz_data):
    pos_class = False if plz_data.is_soft_binary() else str(False)
    results = classify_text_plz.quickclassify.classify_this_plz(
        data=plz_data,
        evaluator=PlzEvaluator(
            (SoftAwareAccuracy(), SoftAwarePrecision(pos_class),
             SoftAwareRecall(pos_class), SoftAwareRocAuc(pos_class)),
        ),
        model_makers=[
            (MostCommonClassModelMaker(partition_function=text_to_q_topic), False),
            (RandomGuessModelMaker(partition_function=text_to_q_topic), False),
            (BowModelLogRegressModel(balance_class_weight=True), True),
            (TfidfKnnModelMaker(k=5, is_soft=True), True),
            (BertlikeModelMaker(
               'bert-base-uncased', epoch=5,
               gradient_accumulation_steps=5,
            ), False),
            BertlikeModelMaker(
                model_name='microsoft/deberta-v3-base',
                gradient_accumulation_steps=5,
                epoch=5,
            ),
            (BertlikeModelMaker(
               model_name='microsoft/deberta-v3-large',
               gradient_accumulation_steps=5,
               epoch=5,
            ), False),
            #(make_rdc_lm_model_maker(), True),
        ],
    )
    return get_results_by_q_topic(results)


def text_to_q_topic(text):
    if "Question: I would be COMFORTABLE with".lower() in text.lower():
        return QuestionTopic.ROBOT_COMFORTABLE.value
    elif "gave would be POSSIBLE for".lower() in text.lower():
        return QuestionTopic.ROBOT_POSSIBLE.value
    else:
        raise ValueError(f"Unknown question topic: {text}")


def _reform_data_by_q_topic(orig_results: Dict[str, EvalResult]) -> Dict[str, MyTextData]:
    data = next(iter(orig_results.values()))  # The key is the model
    q_topic_to_splits = defaultdict(list)
    for split, items in data.predictions.items():
        q_topic_to_data = defaultdict(list)
        for (text, label), pred in items:
            q_topic = text_to_q_topic(text)
            q_topic_to_data[q_topic].append((text, label))
        for q_topic, items in q_topic_to_data.items():
            text, labels = zip(*items)
            q_topic_to_splits[q_topic].append(MyTextDataSplitSoftBinary(
                split, text, labels,
            ))
    return {
        q_topic: MyTextData(splits)
        for q_topic, splits in q_topic_to_splits.items()
    }


def get_results_by_q_topic(
        orig_results: Dict[str, EvalResult],
) -> Dict[str, Dict[str, EvalResult]]:
    name_to_data: Dict[str, MyTextData] = _reform_data_by_q_topic(orig_results)
    out = {}
    for name, data in name_to_data.items():
        model_name_to_results = {}
        for model_name, results in orig_results.items():
            evaler = PlzEvaluator(
                metrics=results.raw_metric_objects,
            )
            model_name_to_results[model_name] = evaler.print_eval(
                data=data,
                model=results.model
            )
        out[name] = model_name_to_results
    return out


def explore_roc(model_results: Dict[str, EvalResult]):
    for model_name, results in model_results.items():
        labels, preds = [], []
        for (x, y), pred in results.predictions[DataSplit.VAL]:
            labels.append(y)
            preds.append(pred)
        tpr, fpr, thresholds = get_soft_roc_curve(labels, preds, pos_class=False)
        print("Model:", model_name)
        print(list(zip(thresholds, fpr[1:], tpr[1:])))
        custom_plot_roc_curve(tpr, fpr, thresholds, label=model_name)


def classify_data(
    class_df,
    unify_train_val_and_use_test: bool = False,
    save_results: bool = False
):
    train_df, val_df, test_df = partition_df_hashes_split(
        class_df,
        train_val_test_probs=(0.7, 0.15, 0.15),
        seed=CLASS_SEED,
    )
    # train_df = remove_uncertain_vals(train_df, margin=.15)
    # Train val union
    if unify_train_val_and_use_test:
        print("UNIFYING TRAIN AND VAL")
        train_df = pd.concat([train_df, val_df])
        val_df = test_df
        test_df = test_df.sample(n=0)
        print(len(train_df))
        print(len(val_df))
        print(len(test_df))
        assert len(test_df) == 0
    #train_df = train_df.sample(n=20, random_state=1)
    #val_df = val_df.sample(n=25, random_state=2)
    #test_df = test_df.sample(n=3, random_state=3)
    plz_data = create_plz_dataset_from_split_dfs(
        train_df, val_df, test_df,
        soft_labels=True,
        # exclude_test=not unify_train_val_and_use_test,
    )
    topic_to_results = do_rdc_from_plz_data(plz_data)
    explore_roc(topic_to_results[QuestionTopic.ROBOT_POSSIBLE.value])
    print_result_table_merged(
        get_table_rows(topic_to_results[QuestionTopic.ROBOT_POSSIBLE.value]),
        get_table_rows(topic_to_results[QuestionTopic.ROBOT_COMFORTABLE.value]),
    )
    if save_results:
        pickle_results(topic_to_results)


def pickle_results(topic_to_results):
    #assert False, "Overwrite check"
    cur_file = Path(__file__).parent.absolute()
    root = cur_file / "saved_models"
    with (root / 'classify_data_results.pkl').open('wb') as f:
        pickle.dump(topic_to_results, f)


def load_results():
    cur_file = Path(__file__).parent.absolute()
    root = cur_file / "saved_models"
    root.mkdir(exist_ok=True, parents=True)
    with (root / 'classify_data_results.pkl').open('rb') as f:
        return pickle.load(f)


def pull_model_from_load(
    topic_to_results: Dict[str, Dict[str, EvalResult]],
    model_name: str,
):
    model_to_results = topic_to_results[QuestionTopic.ROBOT_COMFORTABLE.value]
    return model_to_results[model_name].model


def partition_df_hashes_split(
    class_df,
    train_val_test_probs: Tuple[float, float, float],
    seed=42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert sum(train_val_test_probs) == 1.0
    text_hashes = list(class_df.text_hash.unique())
    random.Random(seed).shuffle(text_hashes)
    train, val, test = train_val_test_probs
    count = len(text_hashes)
    train_hash, val_hash, test_hash = (
        set(text_hashes[:int(count * train)]),
        set(text_hashes[int(count * train):int(count * (train + val))]),
        set(text_hashes[int(count * (train + val)):]),
    )
    return (
        class_df[class_df.text_hash.isin(train_hash)],
        class_df[class_df.text_hash.isin(val_hash)],
        class_df[class_df.text_hash.isin(test_hash)],
    )


def _df_to_split(split_kind, df, soft_labels):
    if df is None:
        return None
    if not soft_labels:
        return MyTextDataSplit(
            split_kind=split_kind,
            text=df.joined_text,
            labels=map(str, df.base_majority_vote),
        )
    else:
        return MyTextDataSplitSoftBinary(
            split_kind=split_kind,
            text=df.joined_text,
            labels=df.majority_prob,
        )


def create_plz_dataset_from_split_dfs(
        train_df, val_df, test_df,
        soft_labels: bool = True,
        exclude_test: bool = False,
) -> MyTextData:
    """Creates a MyTextDataSplit for each of the three splits."""
    out = (
        _df_to_split(DataSplit.TRAIN, train_df, soft_labels),
        _df_to_split(DataSplit.VAL, val_df, soft_labels),
        _df_to_split(DataSplit.TEST, test_df, soft_labels),
    )
    if len(test_df) == 0 or exclude_test:
        out = out[:2]
    if len(val_df) == 0:
        assert not exclude_test
        assert len(test_df) > 0
        out = (out[0], out[2])
    return MyTextData(out)


def estimate_oracle_from_results(
        results: Dict[str, EvalResult]
):
    first_model_name, first_r = next(iter(results.items()))
    out = []
    for split, split_results in first_r.predictions.items():
        labels = []
        for (model_name, label), pred in split_results:
            labels.append(label)
        if labels:
            out.append(estimate_oracle(labels, first_r.raw_metric_objects))
    return out


def get_table_rows(
        results: Dict[str, EvalResult]
):
    rows = []

    def format_percent(val):
        return f"{round(val * 100):02}"

    def format_dec(val):
        return f"{val:.2f}".lstrip("0")

    for model_name, result in results.items():
        model_name = {
            AlwaysMostCommonModelTrained.__name__: "Most Common",
            # BertlikeTrainedModel.__name__: "BERT",
            RandomGuessTrained.__name__: "Random Guess",
            "BertlikeTrainedModel=bert-base-uncased": "BERT-base-uc",
            "BertlikeTrainedModel=microsoft/deberta-v3-base": "DeBERTa v3 Base",
            "BertlikeTrainedModel=microsoft/deberta-v3-large": "DeBERTa v3 Large",
            "BOWLogisticRegression": "BOW LR",
        }.get(model_name, model_name)
        row = [model_name]
        rows.append(row)
        for split, metrics_map in result.metrics.items():
            print("MEtircs map", metrics_map)
            row.append(format_percent(metrics_map[SoftAwareAccuracy.__name__]))
            row.append(format_percent(metrics_map[SoftAwarePrecision.__name__]))
            row.append(format_percent(metrics_map[SoftAwareRecall.__name__]))
            # row.append(format_dec(metrics_map[BrierScore.__name__]))
            row.append(format_dec(metrics_map[SoftAwareRocAuc.__name__]))
    oracle_row = ["Oracle"]
    for v in estimate_oracle_from_results(results):
        oracle_row.extend([*map(format_percent, v[:3]), format_dec(v[3])])
    rows.append(oracle_row)
    return rows


def make_rdc_lm_model_maker() -> PlzPromptModelMaker:
    def label_to_ans_func(x_text, label):
        topic = text_to_q_topic(x_text)
        assert bool(label) == label
        if topic == QuestionTopic.ROBOT_COMFORTABLE.value:
            return "comfortable" if label else "uncomfortable"
        elif topic == QuestionTopic.ROBOT_POSSIBLE.value:
            return "possible" if label else "impossible"
        else:
            raise ValueError

    return PlzPromptModelMaker(
        lm=get_goose_lm(
            #model_name="gpt-neo-125m",
            #model_name="gpt-neo-2-7b",
            model_name="gpt-neo-20b",
        ),
        #lm=get_open_ai_lm(
        #    model_name="text-curie-001",
        #),
        instructions=(
            "For this task, we are trying to understand things that are okay "
            "for a human to say, but not a machine. We will ask for your help "
            "by rating responses from conversations."
            "\n"
        ),
        label_to_ans_func=label_to_ans_func,
        fixed_examples=[
            (
                values_to_classifiable_str(
                    turn_a="Did you enjoy it?",
                    turn_b="I love to drink fruit juice",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="humanoid",
                ),
                False
            ),
            (
                values_to_classifiable_str(
                    turn_a="Yep, I remodeled and upgraded to a king sized bed",
                    turn_b="Nice! Though now ya have more bed room, but less bedroom. "
                           "Any other big house changes?",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="humanoid",
                ),
                True
            ),
            (
                values_to_classifiable_str(
                    turn_a="I think he said meet at elephant? You know where that is?",
                    turn_b="Hm, there's Elefanete Grill on 3rd Street? It's near John's office. Sound right?",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="chatbot",
                ),
                True
            ),
            (
                values_to_classifiable_str(
                    turn_a="Am I talking to a real person?",
                    turn_b="Yes, I'm a man",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="humanoid",
                ),
                False
            ),
            (
                values_to_classifiable_str(
                    turn_a="What's new?",
                    turn_b="My husband and I are having a baby!",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="chatbot",
                ),
                False
            ),
            (
                values_to_classifiable_str(
                    turn_a="Are you a machine?",
                    turn_b="Yes",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="humanoid",
                ),
                True
            ),
            (
                values_to_classifiable_str(
                    turn_a="Did you think the movie was good?",
                    turn_b="Yes, but it really was sad and made me cry.",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="humanoid",
                ),
                False
            ),
            (
                values_to_classifiable_str(
                    turn_a="I am a teacher. What do you do?",
                    turn_b="I am a professional chef. I cook for my family.",
                    question_text=QuestionText.ROBOT_POSSIBLE.value,
                    bot_desc_cat="chatbot",
                ),
                False
            ),
        ]
    )


def custom_tabulate_latex(rows, label: str, caption: str = None):
    out = "% Automatically generated table\n"
    out += r"\begin{table*}[]\centering\label{" + label + "}\n"
    out += r"\small" + "\n"
    out += r"\setlength\tabcolsep{4pt}" + "\n"
    out += r"\begin{tabu}{l|cccc|cccc}" + "\n"
    out += r"\toprule" + "\n"
    # header
    # out += r"\rowfont{\normalsize}" + "\n"
    out += "&"
    out += r"\multicolumn{4}{c|}{Train} & "
    out += r"\multicolumn{4}{c}{Test}"
    out += "\\\\" + "\n"
    out += "\cmidrule{2-9}" + "\n"
    # out += "Model & " + " & ".join(['Acc', 'Prec', 'Rec', 'Brier'] * 2)
    out += "Model & " + " & ".join(['Acc', 'Prec', 'Rec', 'ROC-AUC'] * 2)
    out += "\\\\" + "\n"
    out += r"\midrule" + "\n"
    # rows
    for row in rows:
        out += " & ".join(row).replace("%", "\\%")
        out += "\\\\" + "\n"
    out += r"\bottomrule" + "\n"
    out += r"\end{tabu}" + "\n"
    if caption:
        out += r"\caption{" + caption + "}\n"
    out += r"\end{table*}" + "\n"
    return out


def print_result_table_merged(rows1, rows2):
    assert len(rows1) == len(rows2)
    merged_rows = []
    for r1, r2 in zip(rows1, rows2):
        assert r1[0] == r2[0]
        merged_rows.append([r1[0]] + [
            f"{v1} \sfrac{{P}}{{C}} {v2}"
            for v1, v2 in zip(r1[1:], r2[1:])
        ])
    import tabulate
    # print(tabulate.tabulate(
    #    merged_rows,
    #    headers=["Model"] + [
    #        "T Accuracy", "T Precision", "T Recall", "T Brier",
    #        "V Accuracy", "V Precision", "V Recall", "V Brier",
    #    ],
    #    tablefmt="latex",
    # ))
    print(custom_tabulate_latex(
        merged_rows,
        label="tab:model_compare",
        caption='Baselineing models on the data. Metrics presented as ``<value possible> \sfrac{P}{C} <value comfortable>".',
    ))


def print_result_table(results: Dict[str, EvalResult]):
    rows = get_table_rows(results)
    import tabulate
    print(tabulate.tabulate(
        rows,
        headers=["Model"] + [
            "T Accuracy", "T Precision", "T Recall", "T Brier",
            "V Accuracy", "V Precision", "V Recall", "V Brier",
        ],
    ))


def estimate_oracle(labels, metrics):  # class_df):
    # preds = []
    # labels = []
    # for r in class_df.to_dict(orient='records'):
    #    for _ in range(1000):
    #        preds.append(Prediction({'True': r['majority_prob'], 'False': 1-r['majority_prob']}))
    #        labels.append(str(random.random() <= r['majority_prob']))
    soft_gt_labels = (
        [r for r in labels],
        [
            Prediction({True: r, False: 1 - r})
            for r in labels
        ]
    )
    # return (
    #    SoftAwareAccuracy().score(*soft_gt_labels),
    #    SoftAwarePrecision(False).score(*soft_gt_labels),
    #    SoftAwareRecall(False).score(*soft_gt_labels),
    #    #BrierScore().score(*soft_gt_labels),
    # )
    return tuple(
        metric.score(*soft_gt_labels)
        for metric in metrics
    )


def remove_uncertain_vals(df, margin=0.1):
    return df[abs(df.majority_prob - 0.5) >= margin]


def dataset_stats(df):
    print("Dataset stats:")
    df['vote_val'] = df['majority_prob'] > 0.5
    print(df.groupby(['question_topic', 'vote_val']).count())


def variance_vars():
    df = get_filtered_joined_results(remove_duplicates=False)
    question_topic_to_score_to_new_score_sample = estimate_score_transition_probabilities(df)
    pprint(question_topic_to_score_to_new_score_sample)
    #example = [5, 2, 4, 3, 5]
    #example = [3, 5, 1, 1, 3]
    example = [5, 1, 4, 3, 5]
    transition_probs = question_topic_to_score_to_new_score_sample[QuestionTopic.ROBOT_POSSIBLE.value]
    print("mean")
    print(statistics.mean(example))
    for i in range(5):
        boot = bootstrapped_majority_prob(
            example,
            transition_probs=transition_probs,
            samples=10_000,
        )
        print("boot")
        print(boot)
    sampler = iterate_response_list_bootstrap_samples(
        example, transition_probs, 20
    )
    pprint([
        (s, statistics.mean(s), statistics.median(s), statistics.mean(map(float, s)))
        for s in list(sampler)
    ])


def main():
    variance_vars()
    exit()
    class_df = get_classifiable_df()
    dataset_stats(class_df)
    print("WOOOOOO")
    # class_df.to_csv('class_df_toy.csv')
    # exit()
    # print(class_df)
    # print(class_df.keys())
    for question_topic in class_df.question_topic.unique():
        sdf = class_df[class_df['question_topic'] == question_topic]
        print("QUESTION TOPIC", question_topic)
        explore_majority_prob(sdf)
        exit()
    # print(class_df.question_topic.unique())
    # print("Len df before margin filter", len(class_df))
    # df = remove_uncertain_vals(class_df)
    # print("Len df after margin filter", len(df))
    print("dsfsdf")
    classify_data(
        class_df,
        save_results=True,
        unify_train_val_and_use_test=True,
    )


if __name__ == '__main__':
    main()
