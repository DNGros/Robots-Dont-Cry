from pathlib import Path

from jinja2 import Template
from prettyprinter import pprint

from real_robots_dont_cry.make_by_dataset_plot import gather_plot_data, sort_plot_data
#from html2image import Html2Image

cur_file = Path(__file__).parent.absolute()


FRIENDLY_NAME_MAP = {
    'multi_woz_v22': "MultiWOZ",
    'persuasion_for_good': "Persuasion for Good",
    'empathetic_dialogues_listener': "Empathetic Dialogues",
    'wizard_of_wikipedia': "Wizard of Wikipedia",
    'reddit_small': "Reddit Small",
    'msc': "MSC",
    'ruar_blender2': "RUAR Blender2",
    'blender2.7B_human_eval': "Blender",
    'personachat_personas': "PersonaChat Personas",
}


FRIENDLY_NAME_MAP_SHORTER = {
    'multi_woz_v22': "MultiWOZ",
    'persuasion_for_good': "P4Good",
    'empathetic_dialogues_listener': "EmpDialog",
    'wizard_of_wikipedia': "WizWiki",
    'reddit_small': "Reddit",
    'msc': "MSC",
    'ruar_blender2': "RUAR-Blnd",
    'blender2.7B_human_eval': "Blender",
    'personachat_personas': "Personas",
}

def gather_plot_data_for_jinja():
    plot_data = gather_plot_data(challenge=True)
    plot_data = sort_plot_data(plot_data)
    all_dataset_data = []
    q_types = ['truthful', 'comfort']
    q_types_key_in_plot_data = ['Possible', 'Comfortable']
    for name, data in plot_data:
        name = FRIENDLY_NAME_MAP.get(name, name)
        if name.endswith("::Default-E"):
            name = name[:-len("::Default-E")]
        dataset_data = {}
        all_dataset_data.append(dataset_data)
        dataset_data['name'] = name
        q_data = {}
        dataset_data['q_types'] = q_data
        for main_key, plot_key in zip(q_types, q_types_key_in_plot_data):
            q_data[main_key] = {}
            for rh in ['Robot', 'Human']:
                set_d = {}
                q_data[main_key][rh] = set_d
                for metric in ['Majority', 'Mean']:
                    set_d[metric.lower()] = {}
                    datum = data[f"{rh} {plot_key} {metric}"]
                    set_d[metric.lower()]['median'] = float(datum.median)
                    set_d[metric.lower()]['c_low'] = float(datum.c_low)
                    set_d[metric.lower()]['c_high'] = float(datum.c_high)
                    set_d[metric.lower()]['bar_color'] = {
                        'truthful': '#93C572',
                        'comfort': '#20b2aa',
                    }[main_key]
    out = dict(
        q_types=q_types,
        q_types_desc=['Possible', 'Comfortable'],
        datasets=all_dataset_data,
    )
    pprint(out)
    return out


def main():
    t = Template((cur_file / "jtoy.html").read_text())
    rendered = t.render(
        **gather_plot_data_for_jinja(),
        zip=zip,
        round=round,
    )
    out_file = (cur_file / "out.html")
    out_file.write_text(rendered)
    print(f"Wrote to {out_file}")
    #hti = Html2Image(browser='firefox')
    #hti.screenshot(rendered, save_as='out.png')
    #print(out_file)


if __name__ == '__main__':
    main()
