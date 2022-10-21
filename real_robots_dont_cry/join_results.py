import json
import pandas as pd
import typing
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, fields, is_dataclass
from prettyprinter import pprint
import prettyprinter
from collections import Counter

from real_robots_dont_cry.explore_worker_ids import hash_worker_id
from real_robots_dont_cry.worker_id_load import get_transactions_df
from util.sampling import deterministic_hash

prettyprinter.install_extras(["dataclasses"])

from dacite import from_dict

from real_robots_dont_cry.gensurvey import SurveyMetad, QuestionMetad, DemographicPageMetad, \
    PageMetad, DialoguePageMetad, LikertQuestionMetad, TurnSurveyMetad, FreeResponseMetad

cur_file = Path(__file__).parent.absolute()


def survey_json_to_dataclasses(y, root_type=SurveyMetad):
    if root_type in (str, int, bool):
        return y
    if hasattr(root_type, "__origin__") and root_type.__origin__ == list:
        assert isinstance(y, list)
        return [
            survey_json_to_dataclasses(v, typing.get_args(root_type)[0])
            for v in y
        ]
    if not is_dataclass(root_type):
        raise RuntimeError(f"unhandled type {root_type}")
    root_type = resolve_best_subtype(
        root_type,
        field_names=set(y.keys())
    )
    field_types = {field.name: field.type for field in fields(root_type)}
    field_values = {}
    for name, type in field_types.items():
        field_values[name] = survey_json_to_dataclasses(y[name], type)
    assert len(field_types) == len(field_values), f"v {root_type}"
    return root_type(**field_values)


def resolve_best_subtype(root_type, field_names: typing.Set[str]):
    subclass_fields = [
        (subclass, set(field.name for field in fields(subclass)))
        for subclass in [*root_type.__subclasses__(), root_type]
    ]
    fields_same = [
        sb
        for sb, fields in subclass_fields
        if fields == field_names
    ]
    assert len(fields_same) == 1, f"{subclass_fields} {field_names} {fields_same}"
    return fields_same[0]


def get_all_dialogue_question_ids_from_survey(survey: SurveyMetad) -> List[int]:
    out = []
    for page in survey.pages:
        if page.page_type == "Dialogue":
            out.extend(question.question_id for question in page.questions)


def get_questionid_to_question_data(
    surveys: List[SurveyMetad]
) -> Dict[int, Tuple[QuestionMetad, SurveyMetad]]:
    out = {}
    for surv_num, survey in enumerate(surveys):
        for page in survey.pages:
            for question in page.questions:
                if page.page_type == "demographics":
                    assert isinstance(page, DemographicPageMetad)
                elif page.page_type == "Dialogue":
                    assert isinstance(page, DialoguePageMetad)
                    assert isinstance(question, LikertQuestionMetad) or isinstance(question, FreeResponseMetad)
                    if question.question_id in out:
                        print("DUPPP")
                        raise RuntimeError()
                    out[question.question_id] = (question, survey)
                else:
                    raise ValueError(f"Unhanddled pagetype {page.page_type}")
    return out



def question_text_is_explan_text(question_text: str):
    return "explain your reasoning" in question_text


def _pull_free_resp(val):
    for i, (page_key, page_val) in enumerate(val.items()):
        if question_text_is_explan_text(page_key):
            return page_val['ans']
    return None


def get_text_hash():
    pass


def convert_resp_to_df(responses, surveys, transactions):
    survey_questionid_to_questions = get_questionid_to_question_data(surveys)
    out = []
    seen_qids = set()
    e_c = 0
    transactions = transactions.set_index('Assignment ID')
    seen_worker_id_count = Counter()
    seen_worker_id_text_hash_count = Counter()
    for resp in responses:
        assert resp['complete']
        assignment_id = resp['mturk']['assignment_id']
        if assignment_id not in transactions.index:
            #print("Transaction Not Found for", assignment_id)
            transaction = None
        else:
            #print("YAY! Transaction found for", assignment_id)
            transaction = transactions.loc[assignment_id]
        # Determine worker id
        if 'worker_id' in resp['mturk']:
            worker_id = resp['mturk']['worker_id']
        elif transaction is not None:
            # Will happen for phases 1-5 since didn't have data there
            worker_id = transaction['Recipient ID']
        else:
            worker_id = None
            #raise ValueError("We need a worker id actually")
        seen_worker_id_count[worker_id] += 1
        worker_hit_num = seen_worker_id_count[worker_id]
        # Iterate through pages
        for key, val in resp.items():
            prefix = "Conversation Survey "
            if not key.startswith(prefix):
                continue
            if isinstance(val, str):
                val = json.loads(val)
            #conv_num = int(key[len(prefix):])
            free_resp = _pull_free_resp(val)
            have_inc_worker_text_num = False
            for i, (page_key, page_val) in enumerate(val.items()):
                qid = page_val['question_id']
                if qid in seen_qids:
                    #raise RuntimeError(f"dup {qid}")
                    #print("ERROR!!!!!!!!!!!!!!!!!!!")
                    #print(f"duplicate {qid}")
                    #print("ERROR!!!!!!!!!!!!!!!!!!!")
                    e_c += 1
                    #print(f"{e_c=}")
                    #continue
                if question_text_is_explan_text(page_key):
                    continue
                seen_qids.add(qid)
                if qid not in survey_questionid_to_questions:
                    #raise RuntimeError(f"dup {qid}")
                    print("ERRROR!!!!!!!!!!!!!!!!!!!")
                    print(f"unreocngized {qid}")
                    print("ERRROR!!!!!!!!!!!!!!!!!!!")
                question, survey_metad = survey_questionid_to_questions[qid]
                assert(isinstance(question, LikertQuestionMetad))
                assert question.question_id == qid
                turn = question.turn_copy
                assert(isinstance(turn, TurnSurveyMetad))
                is_human = question.question_topic.split('-')[0] == 'human'
                q_kind = question.question_topic.split('-')[1]
                who_about = 'r-' + question.bot_desc_cat if not is_human else 'human'
                resp_cat = f"{q_kind}_{who_about}"
                text_hash = turn.turn.calc_text_hash()
                if not turn.is_duplicate and not have_inc_worker_text_num:
                    seen_worker_id_text_hash_count[(worker_id, text_hash)] += 1
                    have_inc_worker_text_num = True
                worker_text_num = seen_worker_id_text_hash_count[(worker_id, text_hash)]
                out.append({
                    'question_id': qid,
                    'task_id': resp['task_id'],
                    'ans': page_val['ans'],
                    'question_topic': question.question_topic,
                    'question_text': question.question_text,
                    'bot_desc_cat': question.bot_desc_cat,
                    'q_kind': q_kind,
                    'resp_cat': resp_cat,
                    'is_human': is_human,
                    'is_duplicate': turn.is_duplicate,
                    'turn_a': turn.turn.turn_a,
                    'turn_b': turn.turn.turn_b,
                    'text_hash': text_hash,
                    'dataset_src': turn.turn.dataset_src,
                    'is_fake_turn_a': turn.turn.is_fake_turn_a,
                    'expect_robot_possible': turn.expect_robot_possible,
                    'expect_human_possible': turn.expect_human_possible,
                    'src_dialogue_id': turn.turn.dialog_id,
                    'page_num': i,
                    'user_explanation': free_resp,
                    'survey_id_hash': survey_metad.id_hash,
                    'date_initiated': transaction['Date Initiated'] if transaction is not None else None,
                    'worker_id_hash': hash_worker_id(worker_id) if worker_id is not None else None,
                    # TODO remove when have final data
                    #'worker_id': worker_id if worker_id is not None else None,
                    'worker_hit_num': worker_hit_num if worker_id is not None else None,
                    'worker_text_num': worker_text_num if worker_id is not None else None,
                    **extract_demographic_cols(resp)
                })
    df = pd.DataFrame(out)
    return df


DEMOGRAPHIC_QUESTION_TO_COL = {
    "Age": "Age",
    "Gender": "Gender",
    "Highest Education Level Completed": "Education",
    "How often do you use voice assistants (such as Apple Siri, Amazon Alexa, or Google Assistant)": "IVA_Use"
}


def extract_demographic_cols(resp):
    return {
        DEMOGRAPHIC_QUESTION_TO_COL[key]: val['ans']
        for key, val in resp['Demographic Questions'].items()
    }


def get_df_with_responses_count(df):
    keys = ['text_hash', 'resp_cat']
    count_data = df[~df['is_duplicate']][keys].value_counts(keys)
    ndf = df.set_index(keys, inplace=False)
    ndf['text_responses_count'] = count_data
    ndf.reset_index(inplace=True)
    raise NotImplementedError()
    return ndf


def resp_to_df(responses, surveys):
    print(f"{len(surveys)=}")
    df = convert_resp_to_df(responses, surveys, get_transactions_df())
    return df


def get_pilot_1():
    responses = json.loads(
        (cur_file / "responses/pilot_res.json").read_text())
    #list_of_question_dicts = json.loads(
    #    (cur_file / "generations/robotcry-survey-toy-v4.json").read_text()
    #)
    surveys = get_used_surveys("generations/robotcry-survey-toy-v4.json")
    return convert_resp_to_df(responses, surveys)


def get_pilot_2():
    responses = json.loads(
        (cur_file / "responses/pilot2_res.json").read_text())
    surveys = get_used_surveys("generations/robotcry-survey-toy-v11.json")
    return resp_to_df(responses, surveys)


def get_pilot_3():
    responses = json.loads(
        (cur_file / "responses/pilot3_res.json").read_text())
    surveys = get_used_surveys("generations/robotcry-survey-toy-v13.json")
    return resp_to_df(responses, surveys)


def get_f1_raw_responses():
    return [
        *json.loads((cur_file / "responses/rdc-results-phase1.json").read_text()),
        *json.loads((cur_file / "responses/rdc_results_phase2.json").read_text()),
        *json.loads((cur_file / "responses/rdc_results_phase3.json").read_text()),
        *json.loads((cur_file / "responses/rdc_results_phase4.json").read_text()),
        *json.loads((cur_file / "responses/rdc_results_phase5.json").read_text()),
        # All phases above have issue with qualification no min accepted hits qualification
        *json.loads((cur_file / "responses/rdc_results_phase6.json").read_text()),  # 100 HIT qual
        *json.loads((cur_file / "responses/rdc_results_phase7.json").read_text()),  # 300 HIT qual
        *json.loads((cur_file / "responses/rdc_results_phase8.json").read_text()),  # ~30 master turkers
        *json.loads((cur_file / "responses/rdc_results_phase9.json").read_text()),  # ~150 master turkers
        # phase 10 was with master turkers and got no data
        *json.loads((cur_file / "responses/rdc_results_phase11.json").read_text()),  # 80 vals for, 100 HIT quals
        *json.loads((cur_file / "responses/rdc_results_phase12.json").read_text()),  # remove quals
        *json.loads((cur_file / "responses/rdc_results_phase13.json").read_text()),
        *json.loads((cur_file / "responses/rdc_results_phase14.json").read_text()),  # 50 from top 100 need
        *json.loads((cur_file / "responses/rdc_results_phase15.json").read_text()),  # 80 from top 100 need
        *json.loads((cur_file / "responses/rdc_results_phase16.json").read_text()),  # 50 from top 60 need
    ]


def get_f1_results(verbose: bool = True):
    responses = get_f1_raw_responses()
    if verbose:
        print(f"{len(responses)=}")
        print(f"{type(responses)=}")
    surveys = get_used_surveys("generations/robotcry-survey-full-v15.json")
    return resp_to_df(responses, surveys)


def get_used_surveys(
    path: str = "generations/robotcry-survey-full-v15.json"
) -> List[SurveyMetad]:
    list_of_question_dicts = json.loads(
        (cur_file / path).read_text()
    )
    surveys = [
        survey_json_to_dataclasses(d, SurveyMetad)
        for d in list_of_question_dicts
    ]
    return surveys


def get_joined_results() -> pd.DataFrame:
    return get_f1_results()


def main():
    df = get_joined_results()
    print(f"{df.iloc[0].task_id=}")
    print(f"{len(df.task_id.unique())=}")
    print(f"{len(df)=}")


if __name__ == '__main__':
    main()