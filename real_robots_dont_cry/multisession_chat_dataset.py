from json import JSONDecodeError
from pathlib import Path
from pprint import pprint
import json
from typing import List, Iterable

import parlai
from more_itertools import windowed

from real_robots_dont_cry.gensurvdatas import GenericTurn, text_ok


def extract_msc():
    for ex in _get_msc_exs():
        main_dialogue_texts = [
            d['text'] for d in ex['dialog']
        ]
        all_dialogues = [main_dialogue_texts, *_get_all_prev_dialogues(ex)]
        dialogue_id_base = ex['dialog'][0]['convai2_id']
        for i, d in enumerate(all_dialogues):
            yield from _msc_dialogue_to_generic_turns(d, f"dialogue_id_base_{i}")


def extract_msc_personas():
    all_personas = set()
    for ex in _get_msc_exs():
        for personas in ex['personas']:
            all_personas.update(personas)
    return all_personas


def _get_msc_exs():
    path_parlai = Path(parlai.__path__[0])
    print(path_parlai)
    cur_file = Path(__file__).parent.absolute()
    file = path_parlai / f"../data/msc/msc/msc_dialogue/session_4/train.txt"
    texts = file.read_text().split("\n")
    for t in texts:
        if not t:
            continue
        try:
            ex = json.loads(t)
        except JSONDecodeError as e:
            print(t)
            print(e)
            raise e
        yield ex

def _get_all_prev_dialogues(ex) -> Iterable[str]:
    for d in ex['previous_dialogs']:
        yield [
            t['text']
            for t in d['dialog']
        ]


def _msc_dialogue_to_generic_turns(dialogue: Iterable[str], dialog_id: str) -> Iterable[GenericTurn]:
    for ((a_i, turn_a), (b_i, turn_b)) in windowed(enumerate(dialogue), n=2, fillvalue=(None, None)):
        if turn_b is None:
            continue
        if not text_ok(turn_a, turn_b):
            continue
        yield GenericTurn(
            turn_a,
            turn_b,
            dataset_src="msc",
            dialog_id=dialog_id,
            turn_index=a_i,
            source_metad={
            },
        )


def main():
    msc = list(extract_msc())
    #for turn in msc:
    #    print(turn)
    #print(len(msc))
    all_msc_personas = extract_msc_personas()
    pprint(all_msc_personas)
    print(len(all_msc_personas))


if __name__ == "__main__":
    main()