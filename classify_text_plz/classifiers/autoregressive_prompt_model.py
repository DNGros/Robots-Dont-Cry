from typing import List, Tuple, Callable, Union
import math

from joblib import Memory

from classify_text_plz.classifiers.deeplearn.lmpredictor import LmPrediction, LmPredictor, LmPrompt
from classify_text_plz.dataing import MyTextData
from classify_text_plz.modeling import TextModelMaker, TextModelTrained, Prediction
from classify_text_plz.typehelpers import CLASS_LABEL_TYPE
from classify_text_plz.util import normalize_dict_vals
from functools import lru_cache


def default_formatter_function(x, y) -> str:
    out = f"{x}\nAnswer:"
    if y is not None:
        out += f" {y}"
    return out


class PlzPromptModelMaker(TextModelMaker):
    def __init__(
        self,
        lm: LmPredictor,
        instructions: Union[str, Callable[[str], str]],
        fixed_examples: List[Tuple[str, CLASS_LABEL_TYPE]] = None,
        example_formatter_function: Callable[[str, str], str] = default_formatter_function,
        scorer_function: Callable[[str, LmPrediction], Prediction] = None,
        label_to_ans_func: Callable[[str, CLASS_LABEL_TYPE], str] = None,
        #ans_to_label_func: Callable[[str, CLASS_LABEL_TYPE], CLASS_LABEL_TYPE] = None,
    ):
        self._lm = lm
        self._instructions = instructions
        self._fixed_examples = fixed_examples
        self._example_formatter_function = example_formatter_function
        self._scorer_function = scorer_function
        self._label_to_ans_func = label_to_ans_func
        #self._ans_to_label_func = ans_to_label_func
        #if ((self._label_to_ans_func is None) != (self._ans_to_label_func is None)):
        #    raise ValueError("label_mapper and label_unmapper must be provided together")
        if self._label_to_ans_func is None:
            self._label_to_ans_func = lambda x, y: y
        if fixed_examples is None:
            raise NotImplementedError("learned selected prompts not implemented")

    def fit(self, data: MyTextData) -> TextModelTrained:
        def prompt_maker(txt: str):
            return self._instructions + "\n" + "\n---\n".join(
                self._example_formatter_function(
                    x,
                    self._label_to_ans_func(x, y) if y is not None else None,
                )
                for x, y in [*self._fixed_examples, (txt, None)]
            )
        if self._scorer_function is None:
            self._scorer_function = make_scorer_function(data, self._label_to_ans_func)
        return PlzPromptModel(
            lm=self._lm,
            prompt_maker=prompt_maker,
            scorer_function=self._scorer_function,
        )


def make_scorer_function(
    data,
    label_to_ans_func: Callable[[str, CLASS_LABEL_TYPE], str],
) -> Callable[[str, LmPrediction], Prediction]:
    unique_labels = data.get_unique_labels()
    def scorer(
        x_text: str,
        pred: LmPrediction
    ) -> Prediction:
        tok_to_log_prob = pred.metad['choices'][0]['logprobs']['top_logprobs'][0]
        tok_to_prob = {
            tok: math.exp(log_prob)
            for tok, log_prob in tok_to_log_prob.items()
        }
        option_to_prob = {}
        for tok, prob in tok_to_prob.items():
            tok = tok.strip().lower()
            if tok in option_to_prob:
                option_to_prob[tok] += prob
            else:
                option_to_prob[tok] = prob
        print("option_to_prob", option_to_prob)
        selected = {
            label: option_to_prob.get(
                label_to_ans_func(x_text, label).strip().lower(), 0)
            for label in unique_labels
        }
        print("selected prob", selected)
        if sum(selected.values()) <= 0:
            # Assign equal probability to all
            selected = {
                label: 1 / len(unique_labels)
                for label in unique_labels
            }
        selected = normalize_dict_vals(selected)
        #print("normalized vals", selected)
        return Prediction(selected)

    return scorer

class PlzPromptModel(TextModelTrained):
    def __init__(
        self,
        lm: LmPredictor,
        prompt_maker: Callable[[str], str],
        scorer_function: Callable[[str, LmPrediction], Prediction] = None,
    ):
        self._lm = lm
        self._prompt_maker = prompt_maker
        self._scorer_function = scorer_function

    def predict_text(self, text: str):
        print("predict_text text", text)
        prompt = self._prompt_maker(text)
        print("PROMPT")
        print(prompt)
        print("%%%%%")
        assert prompt.endswith("Answer:")
        assert prompt.strip() == prompt
        result = self._lm.predict(LmPrompt(prompt, max_toks=1, logprobs=25))
        return self._scorer_function(text, result)
