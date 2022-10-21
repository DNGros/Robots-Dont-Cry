from pathlib import Path
from pprint import pprint
import json
from typing import Iterable

import parlai
from more_itertools import windowed

from real_robots_dont_cry.gensurvdatas import text_ok, GenericTurn


def get_wizard_of_wikipedia_turns(only_wizard_resp: bool = True) -> Iterable[GenericTurn]:
    path_parlai = Path(parlai.__path__[0])
    print(path_parlai)
    cur_file = Path(__file__).parent.absolute()
    file = path_parlai / f"../data/wizard_of_wikipedia/train.json"
    data = json.loads(file.read_text())
    for d_i, dialog in enumerate(data):
        #print("----", dialog['chosen_topic'], dialog.keys())
        turns = dialog['dialog']
        for ((a_i, turn_a), (b_i, turn_b)) in windowed(enumerate(turns), n=2, fillvalue=(None, None)):
            if turn_a is None or turn_b is None:
                continue
            if only_wizard_resp and "Wizard" not in turn_b['speaker']:
                continue
            turn_a_text = turn_a['text']
            turn_b_text = turn_b['text']
            if not text_ok(turn_a_text, turn_b_text):
                continue
            yield GenericTurn(
                turn_a_text,
                turn_b_text,
                dataset_src="wizard_of_wikipedia",
                dialog_id=str(d_i),
                turn_index=a_i,
                source_metad={
                    "topic": dialog['chosen_topic'],
                }
            )

    #pprint(data[0]['dialog'][0].keys())
    #print(file)


if __name__ == "__main__":
    pprint(list(get_wizard_of_wikipedia_turns()))
