from pathlib import Path
from pprint import pprint
from collections import Counter
import json

from real_robots_dont_cry.join_results import get_f1_raw_responses

cur_file = Path(__file__).parent.absolute()


def main():
    #data = json.loads((cur_file / "responses/rdc_results_phase7.json").read_text())
    data = get_f1_raw_responses()
    all_free_resps = [
        resp['Feedback'] #if resp['Feedback'] else ""
        for resp in data
    ]
    all_worker_ids = [
        resp['mturk']['worker_id'] if 'worker_id' in resp['mturk'] else None
        for resp in data
    ]
    for worker_id, free_resp in zip(all_worker_ids, all_free_resps):
        print(worker_id, free_resp)
    pprint(Counter(all_free_resps).most_common())


if __name__ == "__main__":
    main()