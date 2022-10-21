import json
from pathlib import Path

from real_robots_dont_cry.gensurvey import SurveyMetad
from real_robots_dont_cry.join_results import survey_json_to_dataclasses

cur_file = Path(__file__).parent.absolute()

if __name__ == "__main__":
    #list_of_question_dicts = json.loads(
    #    (cur_file / "generations/robotcry-survey-toy-v4.json").read_text()
    #)
    responses = json.loads(
        (cur_file / "responses/pilot_res.json").read_text())
    print(f"{len(responses)=}")
    list_of_question_dicts = json.loads(
        (cur_file / "generations/robotcry-survey-toy-v4.1-how.json").read_text()
    )
    surveys = [
        survey_json_to_dataclasses(d)
        for d in list_of_question_dicts
    ]
    print(len(surveys))
