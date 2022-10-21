import statistics
import abc
from typing import Iterable, Sequence, Dict, List, Tuple, Union

from sklearn.metrics import auc

from classify_text_plz.dataing import MyTextData, DataSplit, MyTextDataSplit
from classify_text_plz.modeling import TextModelTrained, Prediction
from classify_text_plz.typehelpers import CLASS_LABEL_TYPE
import numpy as np


class PlzTextMetric(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        expected: Iterable[CLASS_LABEL_TYPE],
        predicted: Iterable[Prediction]
    ) -> float:
        pass

    def __str__(self):
        return self.__class__.__name__

    def metric_name(self):
        return self.__class__.__name__


class PlzTextMetricContinuous(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        expected: Iterable[float],
        predicted: Iterable[Prediction]
    ) -> float:
        pass

    def __str__(self):
        return self.__class__.__name__

    def metric_name(self):
        return self.__class__.__name__


class Accuracy(PlzTextMetric):
    def score(
        self,
        expected: Iterable[CLASS_LABEL_TYPE],
        predicted: Iterable[Prediction]
    ) -> float:
        return statistics.mean(
            int(gt == p.most_probable_label())
            for gt, p in zip(expected, predicted)
        )


def _assert_pred_bool(pred):
    pred_most_prob = pred.most_probable_label()
    if not bool(pred_most_prob) == pred_most_prob:
        raise ValueError(
            f"expect pred labels to be bool, got {pred_most_prob} {type(pred_most_prob)}"
        )


class SoftAwareAccuracy(PlzTextMetricContinuous):
    def score(
        self,
        expected: Iterable[float],
        predicted: Iterable[Prediction]
    ) -> float:
        vals = []
        for gt, p in zip(expected, predicted):
            pred_most_prob = p.most_probable_label()
            _assert_pred_bool(p)
            vals.append(gt if pred_most_prob else 1 - gt)
        return statistics.mean(vals)


class Recall(PlzTextMetric):
    def __init__(
        self,
        pos_class: CLASS_LABEL_TYPE,
    ):
        self._pos_class = pos_class

    def score(
        self,
        expected: Iterable[CLASS_LABEL_TYPE],
        predicted: Iterable[Prediction]
    ) -> float:
        return statistics.mean(
            int(gt == p.most_probable_label())
            for gt, p in zip(expected, predicted)
            if str(gt) == str(self._pos_class)
        )


class SoftAwarePrecision(PlzTextMetricContinuous):
    def __init__(
        self,
        pos_class: bool,
        binary_decision_threshold: float = None
    ):
        assert isinstance(pos_class, bool)
        self._pos_class = pos_class
        self._binary_decision_threshold = binary_decision_threshold

    def score(
        self,
        expected: Iterable[float],
        predicted: Iterable[Prediction]
    ) -> float:
        vals = []
        for gt, p in zip(expected, predicted):
            _assert_pred_bool(p)
            if self._binary_decision_threshold is None:
                pred_most_prob = p.most_probable_label()
            else:
                pred_most_prob = p.get_prob_of(True) > self._binary_decision_threshold
            gt_prob_pos = gt if self._pos_class else 1 - gt
            if pred_most_prob == self._pos_class:
                vals.append(gt_prob_pos)
        return statistics.mean(vals) if len(vals) > 0 else 0


class Precision(PlzTextMetric):
    def __init__(
        self,
        pos_class: CLASS_LABEL_TYPE,
    ):
        self._pos_class = pos_class

    def score(
        self,
        expected: Iterable[CLASS_LABEL_TYPE],
        predicted: Iterable[Prediction]
    ) -> float:
        pred_pos = [
            (gt, p)
            for gt, p in zip(expected, predicted)
            if p.most_probable_label() == self._pos_class
        ]
        if len(pred_pos) == 0:
            return 0
        return statistics.mean(
            int(gt == p.most_probable_label())
            for gt, p in pred_pos
        )


class SoftAwareRecall(PlzTextMetricContinuous):
    def __init__(
        self,
        pos_class: bool,
        binary_decision_threshold: float = None,
    ):
        assert isinstance(pos_class, bool)
        self._pos_class = pos_class
        self._binary_decision_threshold = binary_decision_threshold

    def score(
        self,
        expected: Iterable[float],
        predicted: Iterable[Prediction]
    ) -> float:
        return calc_soft_aware_recall(
            expected, predicted, self._pos_class, self._binary_decision_threshold)


def calc_soft_aware_recall(
    expected: Iterable[float],
    predicted: Iterable[Prediction],
    pos_class: bool,
    binary_decision_threshold: float = None,
):
    vals = []
    gt_pos_sum = 0
    for gt, p in zip(expected, predicted):
        _assert_pred_bool(p)
        if binary_decision_threshold is None:
            pred_most_prob = p.most_probable_label()
        else:
            pred_most_prob = p.get_prob_of(True) > binary_decision_threshold
        gt_prob_pos = gt if pos_class else 1 - gt
        gt_pos_sum += gt_prob_pos
        vals.append(
            gt_prob_pos * (1 if pred_most_prob == pos_class else 0)
        )
    return sum(vals) / gt_pos_sum if gt_pos_sum > 0 else 0


def calc_soft_aware_false_positive_rate(
    expected: Iterable[float],
    predicted: Iterable[Prediction],
    pos_class: bool,
    binary_decision_threshold: float = None,
):
    vals = []
    gt_neg_sum = 0
    for gt, p in zip(expected, predicted):
        _assert_pred_bool(p)
        if binary_decision_threshold is None:
            pred_most_prob = p.most_probable_label()
        else:
            pred_most_prob = p.get_prob_of(True) > binary_decision_threshold
        gt_prob_pos = gt if pos_class else 1 - gt
        gt_neg_prob = 1 - gt_prob_pos
        gt_neg_sum += gt_neg_prob
        vals.append(
            gt_neg_prob * (1 if pred_most_prob == pos_class else 0)
        )
        #if pred_most_prob == pos_class:
        #    vals.append(gt_neg_prob)
    #return statistics.mean(vals) if len(vals) > 0 else 0
    return sum(vals) / gt_neg_sum if gt_neg_sum > 0 else 0


def get_soft_roc_curve(
    expected: Iterable[float],
    predicted: Iterable[Prediction],
    pos_class: bool = True,
):
    expected, predicted = list(expected), list(predicted)
    TPR = [0.0]
    FPR = [0.0]
    thresholds = np.arange(1.00, 0.0, -0.01)
    for th in thresholds:
        if not pos_class:
            th = 1 - th
        TPR.append(calc_soft_aware_recall(
            expected, predicted, pos_class, th
        ))
        FPR.append(calc_soft_aware_false_positive_rate(
            expected, predicted, pos_class, th
        ))
    TPR.append(1.0)
    FPR.append(1.0)
    return TPR, FPR, thresholds


def calc_soft_aware_roc_auc(
    expected: Iterable[float],
    predicted: Iterable[Prediction],
    pos_class: bool = True,
):
    TPR, FPR, thresholds = get_soft_roc_curve(
        expected, predicted, pos_class
    )
    return auc(FPR, TPR)


class SoftAwareRocAuc(PlzTextMetricContinuous):
    def __init__(
        self,
        pos_class: bool,
    ):
        assert isinstance(pos_class, bool)
        self._pos_class = pos_class

    def score(
        self,
        expected: Iterable[float],
        predicted: Iterable[Prediction]
    ) -> float:
        return calc_soft_aware_roc_auc(
            expected, predicted, self._pos_class
        )


def custom_plot_roc_curve(tpr, fpr, thresholds, label=""):
    from matplotlib import pyplot as plt
    #print(fpr)
    #print(tpr)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10,10))
    plt.plot(fpr,tpr,linewidth=2,label='{} (AUC={:.3f})'.format(label, roc_auc))
    plt.plot([0.0,1.0], [0.0,1.0], linestyle='dashed', color='red', linewidth=2, label='random')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("FPR",fontsize=12)
    plt.ylabel("TPR",fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.title("ROC Curve",fontsize=12)
    plt.show()


class BrierScore(PlzTextMetricContinuous):
    def score(
        self,
        expected: Iterable[float],
        predicted: Iterable[Prediction]
    ) -> float:
        return statistics.mean(
            (gt - p.get_prob_of(True)) ** 2
            for gt, p in zip(expected, predicted)
        )


class EvalResult:
    def __init__(
        self,
        model: TextModelTrained,
        metrics: Dict[DataSplit, Dict[str, float]],
        predictions: Dict[DataSplit, List[Tuple[Tuple[str, CLASS_LABEL_TYPE], Prediction]]],
        raw_metric_objects: List[PlzTextMetric],  # hacky since need it later
    ):
        self.model = model
        self.metrics = metrics
        self.predictions = predictions
        self.raw_metric_objects = raw_metric_objects


class PlzEvaluator:
    def __init__(
        self,
        metrics: Sequence[PlzTextMetric] = (Accuracy(), ),
        default_dump_split_highest_loss: Dict[Union[DataSplit, str], int] = None,
    ):
        self.metrics = metrics
        self.default_dump_split_highest_loss = default_dump_split_highest_loss or {
            DataSplit.VAL: 10,
            DataSplit.TEST: 10,
        }

    def print_eval(
        self,
        data: MyTextData,
        model: TextModelTrained,
        #splits: Sequence[DataSplit] = (DataSplit.TRAIN, DataSplit.VAL),
        dump_split_highest_loss: Dict[DataSplit, int] = None,
        dump_split_lowest_loss: Dict[DataSplit, int] = None,
    ) -> EvalResult:
        all_predictions = {split_key: [] for split_key, _ in data.get_all_splits()}
        all_metrics = {split_key: {} for split_key, _ in data.get_all_splits()}
        for split_key, split_data in data.get_all_splits():
            print(f"="*40)
            print("~"*5 + f" Split {split_key}")
            #if split_key in (DataSplit.TRAIN, DataSplit.VAL, DataSplit.TEST):
            #    print("SKIP")
            #    continue
            predictions = [model.predict_text(text) for text in split_data.get_text()]
            for metric in self.metrics:
                if isinstance(metric, PlzTextMetricContinuous):
                    labels = split_data.get_labels()
                else:
                    labels = split_data.get_labels_quantized()
                score = metric.score(labels, predictions)
                print(f"{metric}: {score}")
                all_metrics[split_key][metric.metric_name()] = score
            all_predictions[split_key] = list(zip(split_data.get_text_and_labels(), predictions))
            correct_prob = [
                pred.get_prob_of(gt)
                for gt, pred in zip(split_data.get_labels_quantized(), predictions)
            ]
            #print("CORRECT PROBS", correct_prob)
            correct_prob_and_example = sorted(zip(
                correct_prob,
                split_data.get_text_and_labels(),
                predictions
            ), key=lambda v: v[0])
            if dump_split_highest_loss is None:
                dump_split_highest_loss = self.default_dump_split_highest_loss
            if dump_split_lowest_loss is None:
                dump_split_lowest_loss = {
                    DataSplit.VAL: 3,
                    DataSplit.TEST: 0,
                }
            num_high = dump_split_highest_loss.get(split_key, 0)
            if num_high:
                print(f"-" * 3)
                print(f"Highest {num_high} loss predictions:")
                for prob, (text, gt), pred in correct_prob_and_example[:num_high]:
                    print(f"Correct {prob}: {(text, gt)} "
                          f"pred {pred.most_probable_label()}-{pred.get_prob_of(pred.most_probable_label()):.2f}")
            num_low = dump_split_lowest_loss.get(split_key, 0)
            if num_low:
                print(f"-" * 40)
                print(f"Lowest {num_low} loss predictions:")
                for prob, (text, gt), pred in reversed(correct_prob_and_example[-num_low:]):
                    print(f"Correct {prob}: {(text, gt)} pred {pred.most_probable_label()}")
        return EvalResult(model, all_metrics, all_predictions, self.metrics)

