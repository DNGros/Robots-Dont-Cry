from typing import List, Tuple, Callable

from classify_text_plz.dataing import MyTextData
from classify_text_plz.modeling import TextModelMaker, TextModelTrained


def default_formatter_function(x, y) -> str:
    return f"{x}\nAnswer: {y}"


class PlzOpenAiPromptModelMaker(TextModelMaker):
    def __init__(
        self,
        instructions: str,
        fixed_examples: List[Tuple[str, str]],
        example_formatter_function: Callable[[str], str] = default_formatter_function,
    ):
        self._instructions = instructions
        self._fixed_examples = fixed_examples
        self._example_formatter_function = example_formatter_function

    def fit(self, data: MyTextData) -> TextModelTrained:
        pass


class PlzOpenAiPromptModel(TextModelTrained):
    def __init__(
        self,
        prompt_maker: Callable[[str], str]
    ):
        self._prompt_maker = prompt_maker

    def predict_text(self, text: str):
        pass
