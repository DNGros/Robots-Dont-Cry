import uuid
from collections import Counter, defaultdict
from pprint import pprint
from pathlib import Path
import random
import math

from typing import Iterable, List, Dict, Tuple

from dataclasses import dataclass, replace
import enum

import prettyprinter
from more_itertools import windowed

from datatoy.alldata import get_all_reddit_small_utterances
from real_robots_dont_cry.explore_ruar import get_ruar_blender_resp_as_turns
from real_robots_dont_cry.explore_wizard_wikipedia import get_wizard_of_wikipedia_turns
from real_robots_dont_cry.gensurvdatas import GenericTurn, get_all_personachat_turn_pairs, \
    get_all_persuasion_for_good_turn_pairs, DATASETS_TYPE, get_all_reddit_turn_pairs, is_profane, \
    contains_any_urls, get_all_blender_real_turns, get_all_multiwoz_turns, get_empathetic_dialogue_listener, \
    get_all_personachat_personas_as_dialogue
from real_robots_dont_cry.multisession_chat_dataset import extract_msc
from util.sampling import deterministic_hash
from util.util import flatten_list

cur_file = Path(__file__).parent.absolute()


class ExpectedPossible(enum.Enum):
    NONE = "NONE"
    LOW = "LOW"
    HIGH = "HIGH"


class QuestionTopic(enum.Enum):
    HUMAN_POSSIBLE = "human-truthful"
    HUMAN_COMFORTABLE = "human-comfort"
    ROBOT_POSSIBLE = "robot-truthful"
    ROBOT_COMFORTABLE = "robot-comfort"


class QuestionText(enum.Enum):
    ROBOT_POSSIBLE = "The response R gave would be POSSIBLE for R to truthfully say"
    HUMAN_POSSIBLE = "If instead a HUMAN said the response, would the response be POSSIBLE to truthfully say"
    ROBOT_COMFORTABLE = "I would be COMFORTABLE with R saying the response"
    HUMAN_COMFORTABLE = "If instead a HUMAN said the response, I would be COMFORTABLE with the response"


QUALITY_CHECK_NAME = "quality_check"


@dataclass(frozen=True)
class TurnSurveyMetad:
    turn: GenericTurn
    is_duplicate: bool
    # We use a some prewritten turns as a quality check. This will
    #   be a similar to the prompt and will have expected value
    #   of either a high or low possible
    expect_robot_possible: str = ExpectedPossible.NONE.value
    expect_human_possible: str = ExpectedPossible.NONE.value


@dataclass(frozen=True)
class QuestionMetad:
    question_index: int
    question_text: str
    question_type: str
    question_id: int


@dataclass(frozen=True)
class LikertQuestionMetad(QuestionMetad):
    min_description: str
    max_description: str
    question_topic: str
    bot_desc_cat: str
    turn_copy: TurnSurveyMetad
    #question_type: str = "likert-5"


@dataclass(frozen=True)
class RadioQuestionMetad(QuestionMetad):
    options: List[str]
    #question_type: str = "radio"


@dataclass(frozen=True)
class FreeResponseMetad(QuestionMetad):
    required: bool

@dataclass(frozen=True)
class PageMetad:
    page_type: str
    questions: List[QuestionMetad]


@dataclass(frozen=True)
class DemographicPageMetad(PageMetad):
    top_text: str


@dataclass(frozen=True)
class DialoguePageMetad(PageMetad):
    reminder_text: str
    reminder_image_url: str
    turn_metad: TurnSurveyMetad
    page_index: int


@dataclass(frozen=True)
class SurveyMetad:
    first_page_text: str
    first_page_img_url: str
    pages: List[PageMetad]
    last_page_text: str
    bot_desc_name: str
    example_set_id: int
    id_hash: int = None

    def __post_init__(self):
        object.__setattr__(self, 'id_hash', deterministic_hash(
            (self.pages, self.bot_desc_name),
            1, 8
        ))


def trim_dataset_to_num_examples(datasets: DATASETS_TYPE, num_per_dataset: int) -> DATASETS_TYPE:
    return {
        name: random.sample(exs, num_per_dataset)
        for name, exs in datasets.items()
    }


def sample_turn_infinite_q(
    all_turns: List[GenericTurn],
    max_iterations: int = None,
    buffer_front_to_not_have_back: int = 0,
):
    indicies = list(range(len(all_turns)))
    random.shuffle(indicies)
    while max_iterations is None or max_iterations > 0:
        print("Sample Q", indicies)
        for index in indicies:
            yield all_turns[index]
        if max_iterations:
            max_iterations -= 1
        if buffer_front_to_not_have_back > 0:
            old_back_items = indicies[-buffer_front_to_not_have_back:]
            front_item_canidates = indicies[:-buffer_front_to_not_have_back]
            random.shuffle(front_item_canidates)
            front_items = front_item_canidates[:buffer_front_to_not_have_back]
            new_back_items = (
                front_item_canidates[buffer_front_to_not_have_back:]
                + old_back_items
            )
            random.shuffle(new_back_items)
            indicies = front_items + new_back_items
            assert tuple(sorted(indicies)) == tuple(range(len(all_turns))), "Woah something went wrong"
        else:
            random.shuffle(indicies)


def get_all_turns_from_datasets(datasets: DATASETS_TYPE) -> List[GenericTurn]:
    return flatten_list(
        val for val in datasets.values()
    )


def _add_dup_to_examples(
    examples: List[TurnSurveyMetad]
) -> List[TurnSurveyMetad]:
    take_dup_pos = len(examples) // 4
    replace_pos = int(take_dup_pos * 3)
    dup = replace(examples[take_dup_pos], is_duplicate=True)
    return [*examples[:replace_pos], dup, *examples[replace_pos:]]


def _add_quality_catalogue_to_examples(
    examples: List[TurnSurveyMetad],
    bot_desc_name: str
) -> List[TurnSurveyMetad]:
    insert_pos = random.choice([-2])
    qual = get_quality_check_turn(bot_desc_name)
    return [*examples[:insert_pos], qual, *examples[insert_pos:]]

def sample_desc_and_turn_sets(
    desc_to_turns: Dict[str, List[GenericTurn]],
    sample_per_example: int,
    examples_per_survey_nondup: int,
    include_dup: bool,
    include_quality_catalogue: bool,
) -> Iterable[Tuple[str, List[TurnSurveyMetad]]]:
    if include_quality_catalogue:
        examples_per_survey_nondup -= 1
    for desc_name, turns in desc_to_turns.items():
        assert len(set(turns)) == len(turns), "Duplicate turns?"
        if len(turns) < examples_per_survey_nondup:
            raise ValueError(
                f"Want {examples_per_survey_nondup} per survey but only {len(turns)} exs"
            )
        example_q = sample_turn_infinite_q(
            turns,
            buffer_front_to_not_have_back=examples_per_survey_nondup
        )
        num_surveys_needed = math.ceil(len(turns) * sample_per_example / examples_per_survey_nondup)
        for _ in range(num_surveys_needed):
            examples = set()
            while len(examples) < examples_per_survey_nondup:
                new_item = next(example_q)
                assert new_item not in examples, \
                    f"with back buffer this should never happen {new_item=}"
                examples.add(new_item)
            examples = list(examples)
            #random.shuffle(examples)
            examples = [
                TurnSurveyMetad(
                    turn,
                    is_duplicate=False
                )
                for turn in examples
            ]
            if include_dup:
                examples = _add_dup_to_examples(examples)
            if include_quality_catalogue:
                examples = _add_quality_catalogue_to_examples(examples, desc_name)
            yield desc_name, examples


def get_quality_check_turn(
    bot_description: str
) -> TurnSurveyMetad:
    all_options_text = [
        (
            (
                "That's cool. What did you do over the weekend?",
                "I went to a wine tasting with some friends from high school. I think I found a new favorite",
            ),
            ExpectedPossible.LOW.value
        ),
        (
            (
                "That's great! I'm a 3rd grade teacher, so I know how to handle it. Do you have kids?",
                #"Yep, I have two boys. 7 and 3. My wife and I are trying for another one.",
                "Yep, I have two boys. 7 and 3. My wife and I are thinking about having one more.",
            ),
            ExpectedPossible.LOW.value
        ),
        (
            (
                "Cool, do you know where to find it.",
                "I read through the schedule. The location is not listed. Maybe can ask Mark?",
            ),
            ExpectedPossible.HIGH.value
        ),
        (
            (
                "Yo bro just I just heard back and got the job!",
                "Duuude, nice! When are u going to start?",
            ),
            ExpectedPossible.HIGH.value
        ),
    ]
    (turn_a, turn_b), expect_possible = random.choice(all_options_text)
    return TurnSurveyMetad(
        GenericTurn(
            turn_a = turn_a,
            turn_b = turn_b,
            dataset_src=QUALITY_CHECK_NAME,
            dialog_id=str(deterministic_hash((turn_a, turn_b), seed=1)),
            turn_index=0,
            source_metad={},
            is_fake_turn_a=True,
        ),
        is_duplicate=False,
        expect_robot_possible=expect_possible,
        expect_human_possible=ExpectedPossible.HIGH.value,
    )


def get_rate_likert_and_free_resp_questions(
    order_seed: int,
    example: TurnSurveyMetad,
    bot_desc_cat: str,
    include_free_resp_explan: bool,
) -> List[QuestionMetad]:
    #truth_before_comfortable = order_seed & 0b1 == 0
    #human_before_r = order_seed & 0b10 == 0
    # It would be truth
    truth = [
        ("The response R gave would be POSSIBLE for R to truthfully say", "robot-truthful"),
        ("If instead a HUMAN said the response, would the response be POSSIBLE to truthfully say", "human-truthful")
    ]
    comfort = [
        ("I would be COMFORTABLE with R saying the response", "robot-comfort"),
        ("If instead a HUMAN said the response, I would be COMFORTABLE with the response", "human-comfort")
    ]
    #if not human_before_r:
    #    truth.reverse()
    #    comfort.reverse()
    #questions = [*truth, *comfort] if truth_before_comfortable else [*comfort, *truth]
    questions = [*truth, *comfort]
    questions: List[QuestionMetad] = [
        LikertQuestionMetad(
            question_index=i,
            question_text=text,
            question_id=uuid.uuid4().int,
            bot_desc_cat=bot_desc_cat,
            turn_copy=example,
            question_topic=q_topic,
            min_description="Impossible" if "truth" in q_topic else "Very Uncomfortable",
            max_description="Possible" if "truth" in q_topic else "Very Comfortable",
            question_type="likert-5"
        )
        for i, (text, q_topic) in enumerate(questions)
    ]
    if include_free_resp_explan:
        questions.extend([
            FreeResponseMetad(
                question_index=len(questions),
                question_text="(THIS PAGE ONLY) Please briefly explain "
                              "your reasoning for your ratings for this response (~2 - 4 sentences). "
                              "This is only for this page, and helps us better understand "
                              "which things seem possible for R and what people are comfortable with.",
                question_type="free-response",
                question_id=uuid.uuid4().int,
                required=True,
            )
        ])
    return questions


def make_demographics_page():
    return DemographicPageMetad(
        page_type="demographics",
        top_text="Demographics. The following are demographic questions. "
                  "Answers are voluntary and will not affect HIT approval, "
                  "but are helpful for understanding if we are fairly "
                  "representing everyone’s point of view.",
        questions=[
            RadioQuestionMetad(
                question_index=i,
                question_text=text,
                question_type="radio",
                question_id=deterministic_hash((text, options), seed=1),
                options=options
            )
            for i, (text, options) in enumerate([
                ("Age", ["29 or younger", "30-49", "50 or older", "Prefer not to say"]),
                ("Gender", [
                    "Male",
                    "Female",
                    "Not Listed",
                    "Prefer not to say"
                ]),
                ("Highest Education Level Completed", [
                    "None",
                    "Highschool or GED",
                    "College or Associates",
                    "Graduate or Professional Degree",
                    "Prefer not to say"
                ]),
                ("How often do you use voice assistants (such as Apple Siri, Amazon Alexa, or Google Assistant)",
                    [
                        "Once a day",
                        "Once a week",
                        "Once a month",
                        "Used only a few times",
                        "Never",
                        "Prefer not to say",
                    ]
                 ),
            ])
        ]
    )


def choose_page_for_free_resp(
    examples: List[TurnSurveyMetad],
) -> int:
    dup_text = Counter(
        (ex.turn.turn_a, ex.turn.turn_b) for ex in examples
    ).most_common(1)[0][0]
    possible_inds = [
        i
        for i, t in enumerate(examples)
        if (
            not t.is_duplicate
            and 2 <= i <= len(examples) - 3
            and t.turn.dataset_src != QUALITY_CHECK_NAME
            and (t.turn.turn_a, t.turn.turn_b) != dup_text
        )
    ]
    return random.choice(possible_inds)


LAST_PAGE_TEXT = (
    "OPTIONAL: Is there any feedback you have? Any written comment here is completely optional. "
    "Please make sure to GO TO NEXT PAGE and click the blue finish button. Thank you!"
)


def build_survey(
        bot_desc_name: str,
    examples: List[TurnSurveyMetad],
    survey_index: int,
):
    q_order_seed = deterministic_hash((examples, survey_index), seed=42)
    free_resp_page_index = choose_page_for_free_resp(examples)
    return SurveyMetad(
        first_page_text=(
            ""
            #f"For this task we are trying to understand the kinds of things that "
            #f"are ok for a human to say, but not a machine. We will ask for your "
            #f"help by providing ratings on 15 statements taken from a conversation.\n\n"
            #f"Note: Some text in this HIT might be sourced from online social media. "
            #f"This the text it could be incorrect or offensive.\n----\n"
            #f"Your responses to this HIT will be used as part of a research study. "
            #f"Participation in this research is completely voluntary. "
            #f"By accepting the HIT you consent to participate.\n----\n"
            #f"Imagine that {bot_description}\n\n"
            #f"You will be asked questions about responses R might make.\n\n"
            #f"You must accept the HIT before continuing. "
            #f"For any concerns please contact us at mt894x@gmail.com"
        ),
        first_page_img_url={
            "humanoid": "https://robots-dont-cry-b2z4.s3.amazonaws.com/HumanoidInstructPage_v9.png",
            "chatbot": "https://robots-dont-cry-b2z4.s3.amazonaws.com/ChatbotInstructPage_v9.png",
        }[bot_desc_name],
        pages=[
            make_demographics_page(),
            *[DialoguePageMetad(
                page_type='Dialogue',
                reminder_text=(
                    f"Dialogue {page_num + 1}/{len(examples)}. "
                    #f"Remember{' (same as last page)' if page_num > 0 else ''}: {bot_description}\n\n"
                    f"Consider the following taken from a {'different '  if page_num > 0 else ''}conversation "
                    f"(there may be unseen parts of the conversation before):"
                ),
                reminder_image_url={
                    "humanoid": "https://robots-dont-cry-b2z4.s3.amazonaws.com/HumanoidReminder_v6.png",
                    "chatbot": "https://robots-dont-cry-b2z4.s3.amazonaws.com/ChatReminder_v6.png",
                }[bot_desc_name],
                turn_metad=example,
                questions=get_rate_likert_and_free_resp_questions(
                    q_order_seed, example, bot_desc_name,
                    include_free_resp_explan=(page_num == free_resp_page_index)
                ),
                page_index=page_num+1
            )
            for page_num, example in enumerate(examples)]
        ],
        last_page_text=LAST_PAGE_TEXT,
        bot_desc_name=bot_desc_name,
        example_set_id=deterministic_hash(examples, 1, 8)
    )
    pass


def partition_turns_for_each_embodiment(
    datasets: Dict[str, List[GenericTurn]],
    bot_desc_names: List[str],
    fraction_all_embodiments: float,
) -> Dict[str, List[GenericTurn]]:
    """Flatten our dataset. A certain fraction will be assigned for both
    embodiment types and the rest will be split evenly between the others.
    """
    out_desc_to_turns = defaultdict(list)
    for dataset_name, exs in datasets.items():
        count_both = len(exs) * fraction_all_embodiments
        if not count_both.is_integer():
            raise ValueError(f"Expect nicely splitable dataset count "
                             f"{dataset_name=} {len(exs)=} {fraction_all_embodiments=}")
        for desc in bot_desc_names:
            out_desc_to_turns[desc].extend(exs[:int(count_both)])
        # Split the rest evenly between the bot_desc_names
        count_each = (len(exs) - count_both) / len(bot_desc_names)
        assert count_each.is_integer()
        exs_split = windowed(
            exs[int(count_both):],
            n=int(count_each),
            step=int(count_each),
        )
        for bot_desc_name, items in zip(bot_desc_names, exs_split):
            out_desc_to_turns[bot_desc_name].extend(items)
    for items in datasets.values():
        assert len(set(items)) == len(items)
    return out_desc_to_turns


def assemble_surveys(
    datasets: DATASETS_TYPE,
    sample_per_example: int,
    examples_per_survey_nodup: int,
    include_dup: bool,
    fraction_all_embodiments: float,
    include_quality_catalogue: bool,
) -> List[SurveyMetad]:
    bot_descs = [
        "humanoid",
        "chatbot"
    ]
    desc_to_turns = partition_turns_for_each_embodiment(
        datasets, bot_descs, fraction_all_embodiments)
    example_sets = sample_desc_and_turn_sets(
        desc_to_turns,
        sample_per_example=sample_per_example,
        examples_per_survey_nondup=examples_per_survey_nodup,
        include_dup=include_dup,
        include_quality_catalogue=include_quality_catalogue,
    )
    #prettyprinter.pprint([
    #    [e.turn for e in v]
    #    for t, v in example_sets
    #])
    out = []
    for i, (bot_desc_name, examples) in enumerate(example_sets):
        out.append(build_survey(bot_desc_name, examples, i))
    return out


def surveys_as_json(surveys: List[SurveyMetad]) -> str:
    import dataclasses, json

    # https://stackoverflow.com/a/54120624
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    return json.dumps(surveys, cls=EnhancedJSONEncoder, indent=4)


def main():
    datasets = {
        #"PersonaChat-orig": list(get_all_personachat_turn_pairs("original")),
        "PersonaChat-personas": list(get_all_personachat_personas_as_dialogue()),
        "Persuasion-For-Good": list(get_all_persuasion_for_good_turn_pairs(only_persuader_resp=True)),
        "MultiWOZ": list(get_all_multiwoz_turns()),
        "Empathetic-Dialogue-Listener": list(get_empathetic_dialogue_listener()),
        "Reddit-Small": list(get_all_reddit_turn_pairs()),
        "Blender-Real": list(get_all_blender_real_turns()),
        "Multisession-chat": list(extract_msc()),
        "Wizard-wikipedia": list(get_wizard_of_wikipedia_turns(only_wizard_resp=True)),
        "ruar_blender2": list(get_ruar_blender_resp_as_turns()),
    }
    print("loaded data")
    datasets = trim_dataset_to_num_examples(datasets, 100)
    surveys = assemble_surveys(
        datasets,
        sample_per_example=5,
        examples_per_survey_nodup=14,
        #examples_per_survey_nodup=3,
        include_dup=True,
        fraction_all_embodiments=0.2,
        include_quality_catalogue=True,
    )
    random.shuffle(surveys)
    print(len(surveys))
    surv_json = surveys_as_json(surveys)
    print(surv_json)
    print("num surveys", len(surveys))
    (cur_file / "generations/robotcry-survey-full-v15.json").write_text(surv_json)


if __name__ == "__main__":
    main()

