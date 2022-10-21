from classify_text_plz.dataing import MyTextData
import unicodedata
import nltk


# Actually these functions are kinda dangerous
#def dataset_to_lowercase(data: MyTextData) -> None:
#    data.apply_func_to_text(str.lower)
#
#
#def dataset_to_tokenize(data: MyTextData) -> None:
#    data.apply_func_to_text(
#        lambda s: " ".join(nltk.tokenize.word_tokenize(s))
#    )


def str_to_ascii(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


