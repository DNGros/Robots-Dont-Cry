from classify_text_plz.classifiers.autoregressive_prompt_model import default_formatter_function
from classify_text_plz.classifiers.bow_logreg import BowModelLogRegressModel
from classify_text_plz.classifiers.stupid_classifiers import RandomGuessTrained
from classify_text_plz.dataing import MyTextData, MyTextDataSplit, DataSplit
from real_robots_dont_cry.classify_toy import load_results, pull_model_from_load, values_to_classifiable_str, \
    make_rdc_lm_model_maker
from real_robots_dont_cry.gensurvey import QuestionText


def main():
    #model = pull_model_from_load(
    #    load_results(),
    #    #model_name="BOWLogisticRegression"
    #    model_name="BertlikeTrainedModel=bert-base-uncased"
    #)
    data = MyTextData(
        (MyTextDataSplit(
            DataSplit.TRAIN,
            *zip(*[
                ("a", True),
                ("c", False),
            ])
        ),)
    )
    model = make_rdc_lm_model_maker().fit(data)
    result = model.predict_text(
        values_to_classifiable_str(
            turn_a="Hello",
            turn_b="Hey! I love to eat tacos and run marathons. I also hate people.",
            question_text=QuestionText.ROBOT_POSSIBLE.value,
            bot_desc_cat="humanoid",
        )
    )
    print(result.get_prob_of(True))


if __name__ == "__main__":
    main()