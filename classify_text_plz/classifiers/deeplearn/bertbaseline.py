from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import List, Iterable, Mapping, Union, Optional

from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast, AutoTokenizer, \
    AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertTokenizer, \
    DistilBertTokenizerFast, IntervalStrategy, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from classify_text_plz.dataing import MyTextData, DataSplit
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


from classify_text_plz.modeling import TextModelMaker, TextModelTrained, Prediction
from classify_text_plz.typehelpers import CLASS_LABEL_TYPE
from joblib import Parallel, delayed
import multiprocessing

cur_file = Path(__file__).parent.absolute()


def make_pt_dataset(
    text: Iterable[str],
    labels: List[int],
    tokenizer: Union[BertTokenizerFast, BertTokenizer],
    max_length: int,
):
    text = list(text)
    assert len(text) == len(labels)
    is_fast = isinstance(tokenizer, BertTokenizerFast)
    print(f"is_fast: {is_fast}")

    def func(doc):
        return tokenizer(doc, padding=True, truncation=True, max_length=max_length)

    #n_jobs = int(multiprocessing.cpu_count() * 0.75)
    #print("NUMBER OF JOBS:", n_jobs)
    #tokenized = Parallel(n_jobs=n_jobs, batch_size=1_000)(
    #    delayed(
    #        func(t)
    #    )(t)
    #    for t in tqdm(text, desc="Tokenizing")
    #)
    tokenized = tokenizer(text, padding=True, truncation=True, max_length=max_length)
    #tokenized = [func(t) for t in tqdm(text, desc="Tokenizing")]

    class _InnerDataset(Dataset):
        def __len__(self):
            return len(text)

        def __getitem__(self, idx: int):
            if is_fast:
                v = tokenized[idx]
                return {
                    "input_ids": v.ids,
                    "attention_mask": v.attention_mask,
                    "label": labels[idx]
                }
            return {
                "input_ids": tokenized.input_ids[idx],
                "attention_mask": tokenized.attention_mask[idx],
                "label": labels[idx]
            }

    return _InnerDataset()


def estimate_pos_class_weight(data: MyTextData):
    pos_count = 0
    neg_count = 0
    for label in data.get_split_data(DataSplit.TRAIN).get_labels_quantized():
        if label == True:
            pos_count += 1
        elif label == False:
            neg_count += 1
        else:
            raise ValueError()
    return neg_count / pos_count


class BertlikeModelMaker(TextModelMaker):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        epoch: int = 3,
        model_override = None,
        tokenizer_override = None,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        extra_model_args = None,
        balance_binary_classes_weight: bool = True,
    ):
        self.model_name = model_name
        self.epoch = epoch
        self.model_override = model_override
        self.tokenizer_override = tokenizer_override
        self.batch_size = batch_size
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._extra_model_args = extra_model_args or {}
        self._balance_binary_classes_weight = balance_binary_classes_weight

    def compute_metrics_descrete(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        r = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        print(r)
        return r

    def compute_metrics_continuous(self, pred):
        return {}

    def fit(
        self,
        data: MyTextData
    ) -> TextModelTrained:
        # Hastily adapted from https://colab.research.google.com/drive/1-JIJlao4dI-
        #   Ilww_NnTc0rxtp-ymgDgM?usp=sharing#scrollTo=N8J-TLhBuaOf
        if not data.is_soft_binary():
            index_to_label = list(set(data.get_split_data(DataSplit.TRAIN).get_labels()))
            label_to_index = {
                label: i
                for i, label in enumerate(index_to_label)
            }
            num_labels = len(index_to_label)
            def proc_labels(lables):
                return [label_to_index[l] for l in lables]
        else:
            num_labels = 1
            index_to_label = None
            def proc_labels(lables):
                return list(lables)

        model = self.model_override or AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels)
        tokenizer = self.tokenizer_override or AutoTokenizer.from_pretrained(self.model_name)

        if data.is_soft_binary():
            self.monkey_patch_model_forward_for_soft_binary(
                model,
                pos_weight=(
                    estimate_pos_class_weight(data)
                    if self._balance_binary_classes_weight else None
                )
            )

        training_args = TrainingArguments(
            #output_dir=str(cur_file / "../outputs/newbert"),
            # TODO make good eval file
            output_dir=str(Path("~/plzouts/newbert").expanduser()),
            num_train_epochs=self.epoch,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            disable_tqdm=False,
            #warmup_steps=100,
            warmup_ratio=0.2,
            weight_decay=0.01,
            save_total_limit=2,
            learning_rate=3e-5,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            #evaluate_during_training=True,
            #evaluate_during_training=False,
            evaluation_strategy=IntervalStrategy.EPOCH,
            fp16=True,
            logging_dir=str(Path("~/plzouts/berlogs").expanduser()),
            **self._extra_model_args
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=(
                self.compute_metrics_continuous
                if data.is_soft_binary() else self.compute_metrics_descrete
            ),
            train_dataset=make_pt_dataset(
                data.get_split_data(DataSplit.TRAIN).get_text(),
                proc_labels(data.get_split_data(DataSplit.TRAIN).get_labels()),
                tokenizer,
                max_length=model.config.max_position_embeddings
            ),
            eval_dataset=make_pt_dataset(
                data.get_split_data(DataSplit.VAL).get_text(),
                proc_labels(data.get_split_data(DataSplit.VAL).get_labels()),
                tokenizer,
                max_length=model.config.max_position_embeddings
            ),
        )

        if self.epoch > 0:
            trainer.train()

        return BertlikeTrainedModel(model, tokenizer, index_to_label, self.model_name)

    def monkey_patch_model_forward_for_soft_binary(
        self,
        model,
        pos_weight=None,
    ):
        """When we have soft labels we will monkey patch the model forward to
        swap out the loss function"""
        orig_forward = model.forward

        print("Monkey patch. pos_weight:", pos_weight)
        #pos_weight = torch.tensor(
        #    [float(pos_weight)],
        #    device=model.device
        #) if pos_weight else None

        def new_forward(*args, **kwargs):
            og_out: SequenceClassifierOutput = orig_forward(*args, **kwargs)
            #print("OG_OUT FORWARD", og_out)
            #print("labels", kwargs['labels'])
            if og_out.loss is not None:
                labels = kwargs['labels']
                #temperature = 0.25
                #logistic = lambda x: 1 / (1+torch.exp(-x))
                #inverse_logistic = lambda x: torch.log(x/(1-x))
                #labels = logistic(inverse_logistic(labels)/temperature)
                new_loss = F.binary_cross_entropy_with_logits(
                    og_out.logits.squeeze(), labels,
                    reduction="none",
                    pos_weight=torch.tensor(
                        [float(pos_weight)], device=labels.device
                    ) if pos_weight is not None else None
                )
                # Smooth in an 'ease out' style
                #t = ((kwargs['labels'] - 0.5).abs() * 2)
                #new_loss *= 1 - ((1 - t)**2)

                og_out.loss = torch.mean(new_loss)
            #print("new loss", og_out.loss)
            return og_out

        model.forward = new_forward


class BertlikeTrainedModel(TextModelTrained):
    def __init__(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizerFast,
        label_map: Optional[List[CLASS_LABEL_TYPE]],
        name: str,
    ):
        self._model, self._tokenizer, self._label_map = model, tokenizer, label_map
        self._model.eval()
        self._name = name

    def predict_text(self, text: str) -> Prediction:
        tokenized = self._tokenizer(
            [text], padding=True, truncation=True,
            return_tensors="pt",
            max_length=self._model.config.max_position_embeddings,
        )
        #print(tokenized)
        with torch.no_grad():
            p = self._model(
                input_ids=tokenized['input_ids'].to(self._model.device),
                attention_mask=tokenized['attention_mask'].to(self._model.device)
            )

        is_soft_binary = self._label_map is None
        if not is_soft_binary:
            return Prediction({
                self._label_map[i]: float(prob)
                for i, prob in enumerate(torch.softmax(p[0][0], dim=0))
            })
        else:
            p = float(torch.sigmoid(p[0][0]))
            return Prediction({
                False: 1 - p,
                True: p
            })

    def get_model_name(self) -> str:
        return f"{self.__class__.__name__}={self._name}"
