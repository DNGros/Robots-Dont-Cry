import nltk.translate


def bleu(ref, hyp):
    return nltk.translate.bleu_score.sentence_bleu(
        [ref.lower()],
        hyp.lower(),
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4,
        auto_reweigh=True
    )


if __name__ == "__main__":
    print(bleu(
        "Luguie on cyanide or with rabbies",
        "Luigi is friend with the rabbids"))
