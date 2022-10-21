from real_robots_dont_cry.explore_quality_check import filter_df_by_text_responses_threshold
from real_robots_dont_cry.features import get_features_df
import joblib
from scipy import stats

from real_robots_dont_cry.gensurvey import QuestionTopic


def main():
    df = get_features_df()
    df = filter_df_by_text_responses_threshold(df, 3)
    for q_topic in QuestionTopic:
        print("### " + q_topic.value)
        filt = df[df['question_topic'] == q_topic.value]
        filt = filt.groupby('text_hash').mean()
        for feat in ('sentiment', 'length', 'grammar', 'profanity'):
            print(feat)
            print(stats.spearmanr(filt[feat], filt['ans']))


if __name__ == "__main__":
    main()