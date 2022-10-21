from pathlib import Path
import random
from real_robots_dont_cry.gensurvey import trim_dataset_to_num_examples, assemble_surveys, surveys_as_json
from real_robots_dont_cry.gpt3_exper import load_all_challenge_datasets

cur_file = Path(__file__).parent.absolute()


def main():
    datasets = load_all_challenge_datasets()
    #datasets = trim_dataset_to_num_examples(datasets, 4)
    assert all(len(d) == 40 for name, d in datasets.items())
    surveys = assemble_surveys(
        datasets,
        sample_per_example=35,
        examples_per_survey_nodup=14,
        #examples_per_survey_nodup=2,
        #examples_per_survey_nodup=10,
        include_dup=True,
        fraction_all_embodiments=0.0,
        include_quality_catalogue=True,
    )
    random.shuffle(surveys)
    print(len(surveys))
    surv_json = surveys_as_json(surveys)
    print(surv_json)
    print("num surveys", len(surveys))
    (cur_file / "generations/robotcry-lm-gens-v2.json").write_text(surv_json)


if __name__ == "__main__":
    main()