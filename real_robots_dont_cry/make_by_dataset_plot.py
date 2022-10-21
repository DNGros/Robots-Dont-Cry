from typing import Dict
import numpy as np

from real_robots_dont_cry.data_bootstrap import get_data_bootstrap_sims, filter_sims, ConfIntervalDatum, \
    find_metric_results_for_sim
from matplotlib import pyplot as plt


def gather_plot_data(challenge: bool = False) -> Dict[str, Dict[str, ConfIntervalDatum]]:
    sims = get_data_bootstrap_sims(n_simulations=1000, challenge=challenge)
    #sims = get_data_bootstrap_sims(n_simulations=10)
    #print(sims[0])
    dataset_to_metrics = {}
    print(f"{len(sims)=} {len(sims[0].utterances)=}")
    for dataset in sims[0].all_dataset_srcs:
        filt_sims = filter_sims(sims, dataset_src=dataset)
        dataset_to_metrics[dataset] = find_metric_results_for_sim(filt_sims)
    return dataset_to_metrics


def gather_plot_data_hack_embodiment() -> Dict[str, Dict[str, ConfIntervalDatum]]:
    sims = get_data_bootstrap_sims(n_simulations=20)
    print(sims[0])
    dataset_to_metrics = {}
    print(f"{len(sims)=} {len(sims[0].utterances)=}")
    for dataset in sims[0].all_dataset_srcs:
        for embodiment in sims[0].all_bot_desc_cats:
            filt_sims = filter_sims(sims, dataset_src=dataset, bot_desc_cat=embodiment)
            dataset_to_metrics[f"{dataset}+{embodiment}"] = find_metric_results_for_sim(filt_sims)
    return dataset_to_metrics


def sort_plot_data(data):
    return sorted(
        data.items(),
        key=lambda x: x[1]['Robot Possible Majority'].median,
        reverse=True
    )


def basic_plot_toy():
    data = gather_plot_data()
    sorted_data = sort_plot_data(data)
    #data = gather_plot_data_hack_embodiment()
    #sorted_data = sorted(
    #    data.items(), key=lambda x: x[0], reverse=False)
    #for metric in [
    #    'Robot Possible Majority',
    #    'Human Possible Majority',
    #    'Robot High Comfortable',
    #    'Human High Comfortable',
    #    'Robot Comfortable Majority',
    #    'Human Comfortable Majority',
    #]:
    #    fig, ax = plt.subplots()
    #    ax.set_ylabel('Data Source')
    #    print(data)
    #    ax.set_xlabel(metric)
    #    ax.barh(
    #        width=[val[metric].median for _, val in sorted_data],
    #        y=[dataset for dataset, _ in sorted_data],
    #        xerr=np.array([
    #            [
    #                val[metric].median - val[metric].c_low,
    #                val[metric].c_high - val[metric].median,
    #            ]
    #            for _, val in sorted_data
    #        ]).T,
    #    )
    #    plt.tight_layout()
    #    plt.show()
    #    plt.close()
    #    plt.cla()
    #    plt.clf()
    for metrics in [
        ('Robot Possible Majority', 'Robot Comfortable Majority'),
        ('Human Possible Majority', 'Human Comfortable Majority'),
        ('Robot Comfortable Majority', 'Robot High Comfortable'),
        ('Human Comfortable Majority', 'Human High Comfortable'),
    ]:
        fig, all_ax = plt.subplots(1, len(metrics), sharey=True)
        fig.set_size_inches(10, 5)
        for ax, metric in zip(all_ax, metrics):
            ax.set_ylabel('Data Source')
            print(data)
            ax.set_xlabel(metric)
            ax.barh(
                width=[val[metric].median for _, val in sorted_data],
                y=[dataset for dataset, _ in sorted_data],
                xerr=np.array([
                    [
                        val[metric].median - val[metric].c_low,
                        val[metric].c_high - val[metric].median,
                        ]
                    for _, val in sorted_data
                ]).T,
            )
        # set x limit between 0 and 1 for all plots
        for ax in all_ax:
            ax.set_xlim(0.0, 1.0)
        plt.tight_layout()
        plt.show()
        plt.close()
        plt.cla()
        plt.clf()


def main():
    #df = get_joined_results()
    #df = filter_df_by_pass_quality(df)
    #df = df[~df['is_duplicate'] & ~is_a_quality_check_row(df)]
    #sns.barplot(x='ans', y='dataset_src', data=df)
    #plt.show()
    basic_plot_toy()


if __name__ == '__main__':
    main()
