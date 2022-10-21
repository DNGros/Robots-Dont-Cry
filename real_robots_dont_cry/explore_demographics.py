from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results
from real_robots_dont_cry.gensurvey import QuestionTopic
from real_robots_dont_cry.join_results import get_joined_results, DEMOGRAPHIC_QUESTION_TO_COL
import seaborn as sns
import matplotlib.pyplot as plt


def plot_demographics(df):
    for col in DEMOGRAPHIC_QUESTION_TO_COL.values():
        #ax = sns.barplot(x=col, orient="v", data=df, estimator=lambda x: len(x) / len(df) * 100)
        #ax.set(ylabel="Percent")
        #sns.countplot(x=col, data=df)
        sns.histplot(df, x=col, stat="probability")
        plt.show()


def main():
    #df = get_filtered_joined_results()
    #df = get_joined_results()
    #print(df.keys())
    #print(df.q_kind)
    #df = df[df['page_num'] == 0]
    #df = df[df['question_topic'] == QuestionTopic.ROBOT_COMFORTABLE.value]
    #print(df)
    plot_demographics(df)


if __name__ == "__main__":
    main()
