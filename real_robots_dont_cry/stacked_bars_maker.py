import matplotlib.pyplot as plt
import seaborn as sns


def make_pretty_colors():
    offset = 30
    possible = sns.diverging_palette(
        (360 - (offset / 2)) % 360,
        120 - offset,
        l=70, s=80, as_cmap=False, n=5, center="light")
    comfortable = sns.diverging_palette(
        (360 + (offset / 2)) % 360,
        120 + offset,
        l=70, s=80, as_cmap=False, n=5)
    # Plot the colors
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow([possible, comfortable], aspect='auto', origin='lower')
    plt.show()


def main():
    make_pretty_colors()


if __name__ == "__main__":
    main()