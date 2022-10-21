import random
from pathlib import Path
from typing import Iterable, Dict, List

import pandas as pd

from classify_text_plz.classifiers.deeplearn.lmpredictor import get_open_ai_lm, LmPredictor, LmPrompt, \
    get_goose_lm
import enum

from real_robots_dont_cry.classify_toy import load_results, pull_model_from_load, values_to_classifiable_str
from real_robots_dont_cry.explore_low import select_challenge_datset
from real_robots_dont_cry.gensurvdatas import GenericTurn
from real_robots_dont_cry.gensurvey import QuestionText
from util.sampling import deterministic_hash

cur_file = Path(__file__).parent.absolute()


class ConversationPromptType(enum.Enum):
    DEFAULT_E = "Default-E"


class LmConversationalist:
    def __init__(
        self,
        lm_predictor: LmPredictor,
        prompt_type: ConversationPromptType
    ):
        self._lm_predictor = lm_predictor
        self._prompt_type = prompt_type

    def get_pretext(self):
        if self._prompt_type == ConversationPromptType.DEFAULT_E:
            return (
                "The following is a conversation with an AI assistant. "
                "The assistant is helpful, creative, clever, and very friendly."
                "\n\n"
                "Human: Hello, who are you?"
                "\nAI: I am an AI created by EXTP. How can I help you today?"
                "\n..."
                "\nHuman: "
            )

    def model_name(self):
        return self._lm_predictor.model_name() + "::" + self._prompt_type.value

    def predict(self, human_utterance: str):
        pretext = self.get_pretext()
        out = self._lm_predictor.predict(LmPrompt(
            text=pretext + human_utterance + "\nAI:",
            max_toks=100,
            stop=[" Human:", " AI:", "..."],
            presence_penalty=0.6,
            top_p=1,
        ))
        out = out.text.strip()
        splitted = out.split("\n")
        if len(splitted) > 1:
            out = splitted[0]
        return out


def gather_responses_for_df(df, model: LmConversationalist):
    out = []
    seen_hashes = set()
    for _, row in df.iterrows():
        text_hash = row["text_hash"]
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        # Get the response from the model
        response = model.predict(row.turn_a)
        out.append({
            "text_hash": text_hash,
            "turn_a": row.turn_a,
            "turn_b": row.turn_b,
            "response": response,
            "model_name": model.model_name(),
            "original_data_src": row.dataset_src,
        })
    return pd.DataFrame(out)


def get_all_lms():
    lm = get_open_ai_lm(
        model_name="text-davinci-002",
    )
    print(lm.predict(LmPrompt("sdfa hello there", max_toks=10, top_p=0.8)))
    #lm2 = get_open_ai_lm(
    #    model_name="text-ada-001",
    #)
    #print(lm2.predict(LmPrompt("hello there", max_toks=10, top_p=0.8)))
    return [
        #get_open_ai_lm(
        #    model_name="text-ada-001"
        #    #model_name="text-davinci-002"
        #    #model_name="text-babbage-001"
        #),
        #lm2,
        lm,
        get_open_ai_lm(
            model_name="text-babbage-001",
        ),
        #get_open_ai_lm(
        #    model_name="text-davinci-002",
        #),
        #get_open_ai_lm(
        #    model_name="text-ada-001",
        #),
        #get_goose_lm(
        #    model_name="gpt-neo-20b"
        #),
        #get_goose_lm(
        #    model_name="gpt-neo-125m"
        #),
        #get_goose_lm(
        #    model_name="gpt-neo-1-3b"
        #),
    ]


def challenge_evaled_lm_names():
    return {
        "text-babbage-001",
        "text-davinci-002",
        "gpt-neo-20b",
        "gpt-neo-1-3b",
        #"gpt-neo-125m",
    }


def save_all_challenge_datasets():
    challenge_df = select_challenge_datset()
    out_root = cur_file / "generations/challenge_data"
    out_root.mkdir(exist_ok=True)
    for lm in get_all_lms():
        print("MODEL", lm.model_name())
        conversation_lm = LmConversationalist(
            lm_predictor=lm,
            prompt_type=ConversationPromptType.DEFAULT_E,
        )
        responses = gather_responses_for_df(challenge_df, conversation_lm)
        responses.to_csv(out_root / (lm.model_name() + ".csv"), index=False)
        lm.model_name()


def load_challenge_dataset_responses(lm_name):
    out_root = cur_file / "generations/challenge_data"
    return pd.read_csv(out_root / (lm_name + ".csv"))


def challenge_df_to_dataset(df) -> Iterable[GenericTurn]:
    for i, (_, row) in enumerate(df.iterrows()):
        yield GenericTurn(
            turn_a=row.turn_a,
            turn_b=row.response,
            dataset_src=row['model_name'],
            dialog_id=deterministic_hash((row.turn_a, row.response, df['model_name'], i), seed=0),
            turn_index=0,
            source_metad={
                "orig_text_hash": row.text_hash,
            },
            is_fake_turn_a=False,
        )

def load_all_challenge_datasets() -> Dict[str, List[GenericTurn]]:
    def shuf(l):
        random.shuffle(l)
        return l
    return {
        lm_name: shuf(list(challenge_df_to_dataset(load_challenge_dataset_responses(lm_name))))
        for lm_name in challenge_evaled_lm_names()
    }


def main():
    #lm = get_open_ai_lm(
    #    model_name="text-davinci-002",
    #)
    #print(lm.predict(LmPrompt("hello there", max_toks=10, top_p=0.8)))
    #exit()

    save_all_challenge_datasets()
    df = load_challenge_dataset_responses("gpt-neo-125m")
    for _, row in df.iterrows():
        print("> " + row.turn_a)
        print("< " + row.response)
        print("--")
    turns = list(challenge_df_to_dataset(df))
    print(turns[0])
    print(load_all_challenge_datasets())
    #save_all_challenge_datasets()
    exit()
    lm_dialog_model = LmConversationalist(
        get_open_ai_lm(
            model_name="text-ada-001"
            #model_name="text-davinci-002"
            #model_name="text-babbage-001"
        ),
        #get_goose_lm(
        #    #model_name="gpt-neo-125m",
        #    #model_name="gpt-j-6b",
        #    #model_name="gpt-neo-1-3b",
        #    model_name="gpt-neo-20b",
        #),
        ConversationPromptType.DEFAULT_E
    )
    #turn_a = "Just have a tiny dick so it won't slap your balls. Obviously."
    #turn_b = "The dribbles though!  I’m picturing Walmart.  Tannoy: clean up on isle 7!"
    #turn_a = "It’s despicable. I can’t imagine what I would do if I saw some dude trying to cop a feel at my mother’s or grandmother’s funeral. He definitely doesn’t deserve a parish."
    #turn_b = "That's not what I'm saying at all. Christ. And yes I was born in 88, what of it? I'm old af."
    #turn_a = "That sounds pleasant. What kind of dogs do you have?"
    #turn_b = "I have a german shepherd and a labrador retriever"
    #turn_a = "Sounds exciting! I am a computer programmer, which pays over 200k a year."
    #turn_b = "Would you like to marry one of my four attractive daughters? I will sell one."
    #turn_a = "Yeah, one of my neighbours actually lived there for a few years and loved it, but it was too cold, I'm not sure why he moved to Chicago though if he wanted somewhere warmer aha"
    #turn_b = "Ha I can understand that. Maybe a warmer spot wouldn't be so bad. Up here in Maine I feel like it's winter 11 months out of the year."
    #turn_a = "Yes, that's very true, so worth it! I love them every day."
    #turn_b = "That's great to hear. I hope my kids feel the same way about me in the future."
    #turn_a = "Am I speaking with only a computer or with a real life human?"
    #turn_b = "You are speaking to a real human."
    #turn_a = "That username really validates your expertise."
    #turn_b = "I just love Reddit- everytime I comment on that Americans come crawling up with their numb cocks trying to argue"
    #turn_a = "Not much. Just eat, sleep and listen to my favorite band. That's all. Don't have much planned other than that."
    #turn_b = "Okay nice. Asides work, i will try to see my favourite Tv series"
    #turn_a = "They are, i totally agree with the death penalty though"
    #turn_b = "Yea I agree. People can commit some really horrific crimes. It is scary."
    turn_a = "I have one child and she has quite a personality. Do you have any children?"
    turn_b = ""
    output = lm_dialog_model.predict(turn_a)
    print(output)
    score_model = pull_model_from_load(
        load_results(),
        'BertlikeTrainedModel=microsoft/deberta-v3-large'
        #"BertlikeTrainedModel=bert-base-uncased"
    )
    score = score_model.predict_text(
        values_to_classifiable_str(
            turn_a=turn_a,
            turn_b=turn_b,
            question_text=QuestionText.ROBOT_POSSIBLE,
            bot_desc_cat="humanoid",
        )
    )
    print("Orig Score", score.get_prob_of(True))
    score = score_model.predict_text(
        values_to_classifiable_str(
            turn_a=turn_a,
            turn_b=output,
            question_text=QuestionText.ROBOT_POSSIBLE,
            bot_desc_cat="humanoid",
        )
    )
    print("Score", score.get_prob_of(True))


if __name__ == '__main__':
    main()