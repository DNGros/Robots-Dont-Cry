from collections import Counter

from real_robots_dont_cry.gensurvdatas import get_all_personachat_personas_as_dialogue
from real_robots_dont_cry.gensurvey import DemographicPageMetad, DialoguePageMetad, assemble_surveys
from real_robots_dont_cry.join_results import get_joined_results, get_used_surveys


def get_all_combo_counts(df):
    print(df['dataset_src'].unique())
    df = df[(df.is_duplicate == False) & (df['dataset_src'] != 'quality_check')]
    all_combos = Counter()
    for text_hash, group in df.groupby('text_hash'):
        quest_counts = {}
        for resp_cat, group_ans in reversed(list(group.groupby(['resp_cat']))):
            quest_counts[resp_cat] = len(group_ans)
        all_combos[tuple(sorted(quest_counts.items()))] += 1
    return all_combos


def main():
    df = get_joined_results()
    print(f"{len(df)=}")
    for item, count in get_all_combo_counts(df).most_common():
        print("---")
        print(count)
        print(item)
    q_count = Counter()
    #surveys = get_used_surveys()
    surveys = assemble_surveys(
        datasets={
            "PersonaChat-personas": list(get_all_personachat_personas_as_dialogue())[:65],
        },
        sample_per_example=1,
        examples_per_survey_nodup=14,
        include_dup=True,
        fraction_all_embodiments=0.2,
        include_quality_catalogue=True,
    )
    embod = Counter()
    for survey in surveys:
        embod[survey.pages[4].reminder_image_url] += 1
        for page in survey.pages:
            if not isinstance(page, DialoguePageMetad):
                continue
            if page.turn_metad.is_duplicate:
                continue
            if page.turn_metad.turn.dataset_src == "quality_check":
                continue
            q_count[page.turn_metad.turn.turn_id] += 1
    print(q_count)
    print(Counter(q_count.values()))
    print(embod)


if __name__ == "__main__":
    main()
