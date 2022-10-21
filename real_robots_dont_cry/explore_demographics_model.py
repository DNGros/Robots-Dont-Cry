from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results
from real_robots_dont_cry.gensurvey import QuestionTopic
from real_robots_dont_cry.join_results import get_joined_results, DEMOGRAPHIC_QUESTION_TO_COL
import seaborn as sns
import matplotlib.pyplot as plt


def model_demographics(df):
    # Load packages
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    print(df.columns)
    print(df.question_topic.unique())
    print(df.q_kind.unique())
    print(df.dataset_src.unique())
    df = df[df.dataset_src != 'multi_woz_v22']
    df = df[df.IVA_Use != 'Prefer not to say']
    df = df[df.Education != 'Prefer not to say']
    # Run LMER
    for q_topic in QuestionTopic:
        print(q_topic.name)
        use_df = df[df.question_topic == q_topic.value]
        md = smf.mixedlm(
            "ans ~ Age+Education+IVA_Use+Gender",
            #"ans ~ Age",
            data=use_df,
            groups=use_df["text_hash"],
        )
        mdf = md.fit(method=["lbfgs"])
        print(mdf.summary())


def main():
    df = get_filtered_joined_results()
    model_demographics(df)


if __name__ == "__main__":
    main()
