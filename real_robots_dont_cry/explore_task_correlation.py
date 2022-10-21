import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats

from real_robots_dont_cry.explore_quality_check import get_filtered_joined_results
from real_robots_dont_cry.gensurvey import QUALITY_CHECK_NAME
from real_robots_dont_cry.join_results import get_joined_results
import seaborn as sns


def main():
    df = get_filtered_joined_results()
    df = df[df['dataset_src'] != QUALITY_CHECK_NAME]
    df = df[~df['is_duplicate']]
    #groups = df.groupby(['task_id', 'resp_cat'])
    #mean, std = groups.transform('mean'), groups.transform('std')
    #df['ans'] = (df['ans'] - mean['ans']) / (std['ans'] + 1e-8)
    normalize = True
    if normalize:
        key = 'resp_cat'
        means_stds = df.groupby(key)['ans'].agg(['mean', 'std']).reset_index()
        df = df.merge(means_stds, on=key)
        df['ans_normalized'] = (df['ans'] - df['mean']) / df['std']
    else:
        df['ans_normalized'] = df['ans']
    print(df)
    r_squareds = []
    spear_rs = []
    spear_ps = []
    for task_id in df.task_id.unique():
        print(task_id)
        points = []
        for i, row in df[df['task_id'] == task_id].iterrows():
            relevant = df[(df['text_hash'] == row['text_hash']) & (df['resp_cat'] == row['resp_cat'])]
            other_responses = relevant[relevant['task_id'] != task_id]
            this_response = relevant[relevant['task_id'] == task_id]
            assert len(this_response) == 1
            this_response = this_response.iloc[0]
            for _, other_ans in other_responses.iterrows():
                same_person = df[df['task_id'] == other_ans['task_id']]
                points.append((this_response.ans_normalized, other_ans['ans_normalized']))
        print(points)
        points = np.array(points)
        #sns.regplot(x=points[:, 0], y=points[:, 1])
        #plt.show()
        #sns.swarmplot(x=points[:, 0], y=points[:, 1])
        #plt.show()
        slope, intercept, r_value, p_value, std_err = stats.linregress(points[:, 0], points[:, 1])
        print(f"{r_value=:.3f} {p_value=:.3f} {slope=:.3f} {intercept=:.3f}")
        r_squareds.append(r_value ** 2)
        sprearman_r, spearman_p = stats.spearmanr(points[:, 0], points[:, 1])
        print(f"{sprearman_r=:.3f} {spearman_p=:.3f}")
        spear_rs.append(sprearman_r)
        spear_ps.append(spearman_p)
        #exit()
    print(f"{np.mean(r_squareds)=:.3f} {np.median(r_squareds)=:.3f}")
    print(f"spearman r: {np.mean(spear_rs)=:.3f} {np.median(spear_rs)=:.3f}")
    print(f"spearman ps: {np.mean(spear_ps)=:.3f} {np.median(spear_ps)=:.3f}")
    sns.histplot(r_squareds)
    plt.title("R-squared")
    plt.show()
    sns.histplot(spear_rs)
    plt.title("Spearman Rs")
    plt.show()
    sns.histplot(spear_ps)
    plt.title("Spearman Ps")
    plt.show()


if __name__ == "__main__":
    main()
