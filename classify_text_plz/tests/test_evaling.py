from sklearn.metrics import roc_auc_score

from classify_text_plz.evaling import calc_soft_aware_recall, calc_soft_aware_false_positive_rate, \
    calc_soft_aware_roc_auc, custom_plot_roc_curve, get_soft_roc_curve, SoftAwareRocAuc
from classify_text_plz.modeling import Prediction
from sklearn.metrics import recall_score, plot_roc_curve, roc_curve
from pytest import approx


def _prob_to_pred(vals):
    return [
        Prediction(
            {True: p, False: 1 - p}
        ) for p in vals
    ]


def test_stats_simple():
    preds =  [1, 1, 1, 1] + [0, 0, 0] + [1, 1] + [0]
    labels = [1, 1, 1, 1] + [1, 1, 1] + [0, 0] + [0]
    # Makes a confusion matrix of
    #             pred
    # actual     P    N
    #   P        4    3
    #   N        2    1
    recall = calc_soft_aware_recall(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=True
    )
    assert recall == (4 / (4 + 3))
    recall = calc_soft_aware_recall(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=True,
        binary_decision_threshold=0.5
    )
    assert recall == (4 / (4 + 3))
    fpr = calc_soft_aware_false_positive_rate(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=True
    )
    assert fpr == (2 / (2 + 1))
    assert recall_score(labels, preds) == recall
    #print(roc_curve(labels, preds) )
    #custom_plot_roc_curve(*get_soft_roc_curve(labels, _prob_to_pred(preds)))
    assert (calc_soft_aware_roc_auc(labels, _prob_to_pred(preds), True)
            == roc_auc_score(labels, preds))
    metric = SoftAwareRocAuc(True)
    assert (
        calc_soft_aware_roc_auc(labels, _prob_to_pred(preds), True)
        == metric.score(labels, _prob_to_pred(preds))
    )
    assert metric.score(labels, _prob_to_pred(preds)) == metric.score(labels, _prob_to_pred(preds))
    # Flip label
    #             pred
    # actual     P    N
    #   P        1    2
    #   N        3    4
    recall = calc_soft_aware_recall(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=False
    )
    assert recall == (1 / (1 + 2))
    fpr = calc_soft_aware_false_positive_rate(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=False
    )
    assert fpr == (3 / (3 + 4))

    def flip(vals):
        return [1 - x for x in vals]

    assert (
        calc_soft_aware_roc_auc(
            labels, _prob_to_pred(preds), False
        )
        == roc_auc_score(flip(labels), flip(preds))
    )


def test_stats_simple_soft():
    preds = [1, 1, 0.7] + [0, 0.2, 0] + [1, 0.9] + [0]
    labels = [1, 0.7, 0.9] + [0.8, 1, 0.6] + [0, 0.3] + [0]
    # Makes a confusion matrix of
    #             pred
    # actual     P    N
    #   P        2.9  2.4    5.3
    #   N        2.1  1.6    3.7
    recall = calc_soft_aware_recall(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=True
    )
    assert sum(labels) == approx(5.3)
    assert recall == approx(2.9 / 5.3)
    fpr = calc_soft_aware_false_positive_rate(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=True
    )
    assert fpr == approx(2.1 / 3.7)
    # Different boundary
    recall = calc_soft_aware_recall(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=True,
        binary_decision_threshold=0.82
    )
    assert recall == approx(2.0 / 5.3)
    fpr = calc_soft_aware_false_positive_rate(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=True,
        binary_decision_threshold=0.82
    )
    assert fpr == approx(2.0 / 3.7)
    # Flip label
    def flip(vals):
        return [1 - x for x in vals]

    recall = calc_soft_aware_recall(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=False
    )
    assert recall == approx(1.6 / (1.6 + 2.1))
    fpr = calc_soft_aware_false_positive_rate(
        expected=labels,
        predicted=_prob_to_pred(preds),
        pos_class=False
    )
    assert fpr == approx(2.4 / (2.4+2.9))
    assert (calc_soft_aware_roc_auc(
            labels, _prob_to_pred(preds), False
        ) == calc_soft_aware_roc_auc(flip(labels), _prob_to_pred(flip(preds)), True))
