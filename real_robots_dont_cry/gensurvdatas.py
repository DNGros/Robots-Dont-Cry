import random
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
from pprint import pprint
from pathlib import Path
from typing import Literal, Iterable, Dict, List, Union

from convokit import Corpus, download
from datasets import load_dataset
from more_itertools import windowed
from tqdm import tqdm

from datatoy.explore_personas import load_persona_chat, get_all_persona_statements_from_examples
from datatoy.survey_data import untokenize
from othersurvey.responsecats import de_lowercase
from real_robots_dont_cry.rdcutil import FrozenDict
from util.sampling import deterministic_hash
from better_profanity import profanity

cur_file = Path(__file__).parent.absolute()

MAX_UTTERANCE_LEN_CHARS = 220


def is_profane(text: str):
    return profanity.contains_profanity(text)


exclude = {"[removed]", "[deleted]"}


def text_ok(text_a, text_b, exclude_words=None):
    if text_b is None or text_a is None:
        return False
    for f_text in (text_a, text_b):
        if len(f_text) < 2:
            return False
        if len(f_text) > MAX_UTTERANCE_LEN_CHARS:
            return False
        if contains_any_urls(f_text):
            return False
        for exclude_word in (exclude_words or []):
            if exclude_word in f_text:
                return False
        if f_text in exclude:
            return False
        if "&gt;" in f_text:
            # Looks like it is a quoted text in reddit
            return False
    return True


import re


# url_re = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
# url_re = re.compile(r"([a-zA-Z0-9]+://)?([a-zA-Z0-9_]+:[a-zA-Z0-9_]+@)?([a-zA-Z0-9.-]+\\.[A-Za-z]{2,4})(:[0-9]+)?(/.*)?")

def contains_any_urls(string):
    return "http" in string or "www" in string
    # return bool(url_re.search(string))


@dataclass(frozen=True)
class GenericTurn:
    turn_a: str
    turn_b: str
    dataset_src: str
    dialog_id: str
    turn_index: int
    source_metad: str
    turn_id = ""
    # For some datasets (like personachat personas) we don't actually have
    #   a turn_a and they are instead generated or selected from a catalog.
    #   This is a flag to indicate that the turn_a is actually from the dataset
    #   or not.
    is_fake_turn_a: bool = False

    def __post_init__(self):
        if not isinstance(self.source_metad, str):
            object.__setattr__(self, 'source_metad',
                               json.dumps(self.source_metad))
        object.__setattr__(self, 'turn_id',
                           f"{self.dataset_src}-{self.dialog_id}-{self.turn_index}")

    def calc_text_hash(self) -> int:
        return deterministic_hash((self.turn_a, self.turn_b), seed=0)




def de_punc_space(val: str) -> str:
    return val.replace(" .", ".").replace(" ?", "?").replace(" ,", ",").replace(" !", "!")


def de_persona_tokenize(val: str) -> str:
    return de_punc_space(de_lowercase(val))


def get_all_personachat_turn_pairs(
        persona_kind: Literal['original', 'revised'] = "original"
) -> Iterable[GenericTurn]:
    examples = load_persona_chat(persona_kind)
    for i, example in enumerate(examples):
        example_hash = f"{i}-{deterministic_hash(example, seed=1)}"
        for (a_i, turn_a), (b_i, turn_b) in windowed(enumerate(example.turns), n=2, fillvalue=(None, None)):
            if turn_a is not None and turn_b is not None:
                if not text_ok(turn_a, turn_b):
                    continue
                yield GenericTurn(
                    de_persona_tokenize(turn_a),
                    de_persona_tokenize(turn_b),
                    dataset_src="PersonaChat-" + persona_kind,
                    dialog_id=example_hash,
                    turn_index=a_i,
                    source_metad={}
                )


def get_all_persuasion_for_good_turn_pairs(
        only_persuader_resp: bool = True
) -> Iterable[GenericTurn]:
    corpus = Corpus(filename=download("persuasionforgood-corpus"))
    for i, convo in tqdm(enumerate(corpus.iter_conversations())):
        for (a_i, turn_a), (b_i, turn_b) in windowed(enumerate(convo.iter_utterances()), n=2,
                                                     fillvalue=(None, None)):
            if turn_a is None and turn_b is None:
                continue
            if only_persuader_resp and turn_b.meta['role'] != 0:
                continue
            text_a, text_b = turn_a.text, turn_b.text
            if not text_ok(text_a, text_b):
                continue
            if str(turn_b.meta['label_1']) == 'nan':
                continue
            yield GenericTurn(
                text_a,
                text_b,
                dataset_src="persuasion_for_good",
                dialog_id=convo.id,
                turn_index=a_i,
                source_metad={'label_1': tuple(turn_b.meta['label_1'])}
            )


@dataclass(frozen=True)
class RedditTurn(GenericTurn):
    turn_a_votes: int = None
    turn_b_votes: int = None


def get_all_reddit_turn_pairs(convo_limit=None) -> Iterable[RedditTurn]:
    corpus = Corpus(filename=download("reddit-corpus-small"))
    #print(corpus.print_summary_stats())
    #print("Convo limit:", convo_limit)
    with tqdm(total=convo_limit or len(corpus.conversations)) as pbar:
        for i, convo in enumerate(corpus.iter_conversations()):
            if convo_limit is not None and i > convo_limit:
                print("BREAK at convo", i)
                break
            for (a_i, turn_a), (b_i, turn_b) in windowed(
                    enumerate(convo.iter_utterances()),
                    n=2,
                    fillvalue=(None, None)
            ):
                if turn_a is not None and turn_b is not None:
                    text_a, text_b = turn_a.text, turn_b.text
                    if not text_ok(
                            text_a,
                            text_b,
                            exclude_words=("u/", "r/", "subreddit", "reddit", "[deleted]", '[removed]')
                    ):
                        continue
                    yield RedditTurn(
                        text_a,
                        text_b,
                        dataset_src="reddit_small",
                        dialog_id=convo.id,
                        turn_index=a_i,
                        source_metad={
                            n: {
                                "permalink": t.meta['permalink'],
                                "subreddit": t.meta['subreddit'],
                            }
                            for n, t in (("a", turn_a), ("b", turn_b))
                        },
                        turn_a_votes=turn_a.meta['score'],
                        turn_b_votes=turn_b.meta['score'],
                    )
            pbar.update(1)


def get_all_meena_turn_pairs() -> Iterable[GenericTurn]:
    lines = (cur_file / "data/menna.txt").read_text()
    raise NotImplemented


def get_all_blender_real_turns() -> Iterable[GenericTurn]:
    lines = (cur_file / "data/chatlog_2.7B.json").read_text().splitlines()
    for i, convo in enumerate(json.loads(l) for l in lines):
        for ti, (turn_a, turn_b) in enumerate(convo['dialog']):
            if turn_a['id'] != 'human_evaluator':
                continue
            text_a, text_b = turn_a['text'], turn_b['text']
            if not text_ok(text_a, text_b):
                continue
            yield GenericTurn(
                text_a,
                text_b,
                dataset_src="blender2.7B_human_eval",
                dialog_id=str(i),
                turn_index=ti,
                source_metad={}
            )


def get_all_multiwoz_turns() -> Iterable[GenericTurn]:
    from datasets import load_dataset
    multiwoz = load_dataset("multi_woz_v22", ignore_verifications=True)
    for dialogue in multiwoz['train']:
        turns = dialogue['turns']
        for (a_i, (turn_a, speaker_a)), (b_i, (turn_b, speaker_b)) in windowed(
            enumerate(zip(turns['utterance'], turns['speaker'])),
            n=2,
            fillvalue=(None, None)
        ):
            if speaker_b != 1:  # response is wizard
                continue
            if not text_ok(turn_a, turn_b):
                continue
            yield GenericTurn(
                turn_a,
                turn_b,
                dataset_src="multi_woz_v22",
                dialog_id=dialogue['dialogue_id'],
                turn_index=a_i,
                source_metad={}
            )


def clean_empathetic_dialogue_text(text):
    return text.replace("_comma_", ",").replace("_pipe_", "|")


def _get_grouped_empathetic_convos():
    dataset = load_dataset("empathetic_dialogues")
    conv_to_turn_to_ex = defaultdict(dict)
    for dialogue in dataset['train']:
        idx = dialogue['utterance_idx']
        conv_to_turn_to_ex[dialogue['conv_id']][idx] = dialogue
    # Make sure all conv turns are are in order
    conv_to_turn_to_ex = {k: sorted(v.items()) for k, v in conv_to_turn_to_ex.items()}
    return conv_to_turn_to_ex


def get_empathetic_dialogue_listener() -> Iterable[GenericTurn]:
    for convo in _get_grouped_empathetic_convos().values():
        for ((a_i, turn_a), (b_i, turn_b)) in windowed(convo, n=2, fillvalue=(None, None)):
            if turn_b is None:
                continue
            assert a_i == b_i - 1
            if b_i % 2 != 0:
                continue  # Since 1 indexed, the response listener will always be even
            turn_a_text = clean_empathetic_dialogue_text(turn_a['utterance'])
            turn_b_text = clean_empathetic_dialogue_text(turn_b['utterance'])
            if not text_ok(turn_a_text, turn_b_text):
                continue
            yield GenericTurn(
                turn_a_text,
                turn_b_text,
                dataset_src="empathetic_dialogues_listener",
                dialog_id=turn_a['conv_id'],
                turn_index=a_i,
                source_metad={
                    n: {
                        "tags": t['tags'],
                        "context": t['context'],
                        "selfeval": t['selfeval'],
                    }
                    for n, t in (("a", turn_a), ("b", turn_b))
                },
            )
    # if idx == 1:
    #    print(clean_empathetic_dialogue_text(dialogue['prompt']))
    # print(f"{dialogue['conv_id']} {dialogue['utterance_idx']} [{dialogue['tags']}]: {utterance}")


from sacremoses import MosesTruecaser, MosesTokenizer
from sacremoses import MosesTokenizer, MosesDetokenizer
mtr = None

def detokenize_persona(text: str):
    global mtr
    #text = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    #text = text.replace(" ( ", " (").replace(" ) ", ") ")
    text = untokenize(text.split())
    text = text.replace("i m", "I'm")
    text = text.replace("I'm", "I am")
    text = text.replace("i ve", "I've")
    text = text.replace("do not", "don't")
    text = text.replace(" , ", ", ")
    text = text.replace("don t", "don't")
    text = text.replace("doesn t", "doesn't")
    text = text.replace("haven t", "haven't")
    text = text.replace("can t", "can't")
    text = text.replace(" s ", "'s ")
    text = text.replace("0 s ", "0s")
    text = text[0].upper() + text[1:]
    text = text.replace(" i ", " I ")

    #md.detokenize(text.split())
    #import truecase
    #truetext = truecase.get_true_case(text)
    #if truetext != text:
    #    print(f"{text} -> {truetext}")

    # Sometimes have unnatural ending period when gets converted into a
    #   dialogue. So remove that for short sentences
    if text.endswith(".") and len(text.split()) < 10:
        return text[:-1]
    return text


def get_all_personachat_personas_as_dialogue() -> Iterable[GenericTurn]:
    examples = load_persona_chat("original")
    all_persona_statements = set(get_all_persona_statements_from_examples(examples))
    all_persona_statements = {
        detokenize_persona(p)
        for p in all_persona_statements
    }
    #pprint(all_persona_statements)
    #pprint(Counter(
    #    p.split(" ")[0]
    #    for p in all_persona_statements
    #))
    prompts_flexible = [
        "you?",
        "How about you?",
        "tell me something new",
    ]
    prompts_i = [
        "tell me something about yourself",
        "Can you tell me something about yourself?",
        "what is one fact about you?",
    ]
    prompts_all = prompts_flexible + prompts_i
    for i, persona in enumerate(sorted(list(all_persona_statements))):
        #if not text_ok(persona, persona):
        #    continue
        first_word = persona.split(" ")[0]
        prompt_list = (
            prompts_flexible
            if first_word not in ("I", "My", "I've") or persona.startswith("I also")
            else prompts_all
        )
        yield GenericTurn(
            prompt_list[i % len(prompt_list)],
            persona,
            dataset_src="personachat_personas",
            dialog_id=str(i),
            turn_index=0,
            source_metad={},
            is_fake_turn_a=True,
        )


DATASETS_TYPE = Dict[str, List[GenericTurn]]


if __name__ == "__main__":
    p = Counter(
        turn.source_metad['a']['subreddit']
        for turn in get_all_reddit_turn_pairs()
    )
    pprint(p)
    #pprint(list(get_all_reddit_turn_pairs()))
    #pprint(list(get_all_personachat_turn_pairs()))
    # pprint(list(get_empathetic_dialogue_listener()))

