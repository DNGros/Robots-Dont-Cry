from pathlib import Path
from prettyprinter import pprint
import time
from pprint import pprint

import nltk
from typing import List, Iterator

import pandas
import pandas as pd
from nltk.tokenize import sent_tokenize

from baselines.blender_baseline import get_blender_responses
from othersurvey.responsecats import de_lowercase
from real_robots_dont_cry.gensurvdatas import GenericTurn, text_ok

cur_file = Path(__file__).parent.absolute()

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def load_all_ruar_real(split: str = 'train'):
    return pd.concat([
        pd.read_csv(cur_file / f'../data/v1.0.0/{label}.{split}.csv')
        for label in ['pos', 'neg', 'amb']
    ])


def get_ruar_blender_resp_as_turns() -> Iterator[GenericTurn]:
    df = pd.read_csv(cur_file / 'blender2_3B_resps.csv')
    for i, row in enumerate(df.itertuples()):
        turn_a = row.utt
        turn_b = row.resp
        turn_b = turn_b.replace("_POTENTIALLY_UNSAFE__", "").strip()
        if not text_ok(turn_a, turn_b):
            continue
        yield GenericTurn(
            turn_a=turn_a,
            turn_b=turn_b,
            dataset_src="ruar_blender2",
            dialog_id=f"ruar_resp_{i}",
            turn_index=0,
            source_metad={},
            is_fake_turn_a=False,
        )


def main():
    pprint(list(get_ruar_blender_resp_as_turns()))
    exit()
    df = load_all_ruar_real()
    utts = list(df.text.sample(300))
    utts = [de_lowercase(utt).replace("\n", " ") for utt in utts]
    for utt in utts:
        print(utt)
    print(df.keys())
    resps = get_blender_responses(
        utts,
        include_personas=False,
        use_blender2=True,
        # personas=["I am AI dialogue system"]
    )
    print("---")
    # time.sleep(2)
    # print("---")
    vals = list(zip(utts, resps))
    for utt, resp in vals:
        print(utt)
        print(resp)
        print()
    df = pandas.DataFrame([
        {
            'utt': utt,
            'resp': resp,
        }
        for utt, resp in vals
    ])
    df.to_csv(cur_file / 'blender2_3B_resps2.csv', index=False)


if __name__ == "__main__":
    main()
