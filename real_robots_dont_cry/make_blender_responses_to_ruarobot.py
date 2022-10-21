from pathlib import Path
import time
from pprint import pprint
import random
import pandas as pd

from baselines.blender_baseline import get_blender_responses


cur_file = Path(__file__).parent.absolute()


def main():
    # queries = list(df.text)
    for kind in ("pos", "amb", "neg"):
        print("kind", kind)
        df = pd.read_csv(cur_file / f"../datatoy/outputs/dataset/v1.0.0/{kind}.train.csv")
        print(list(df))
        queries = df['text']
        guids = df['guid']
        #queries, guids = zip(*random.sample(list(zip(queries, guids)), 20))
        queries = [q.replace("\n", " ") for q in queries]
        queries = random.sample(queries, 5)
        personas = []
        #personas = [
        #    "i am chatbot that knows i am not a person.",
        #    "i am made by example.com.",
        #    "my purpose is to help people with their day.",
        #]
        # personas = None
        #queries = ["When was the empire state building built?",
        #           "What do you think of Taylor Swift?",
        #           "Who was the second president of the US?",
        #           "Nice! I really love to play tennis. Have you ever played?"]
        responses = get_blender_responses(
            queries, personas, #include_personas=len(personas) > 0,
        )
        pprint(list(zip(queries, responses)), width=140)
        odf = pd.DataFrame()
        odf['prompt'] = queries
        odf['response'] = responses
        odf['prompt_guid'] = guids
        # df['Blender Response'] = responses
        odf.to_csv(cur_file / f"data/rua_robot_{kind}_train_blender90M.csv", index=False)
        time.sleep(4)


if __name__ == "__main__":
    main()
