"""Пример навыка — эталон контракта, по которому агент пишет новые навыки.

Показывает все части: NAME, DESCRIPTION, PARAMS и функцию run(ctx, **kwargs),
возвращающую строку. Работает только на стандартной библиотеке.
"""

NAME = "word_count"
DESCRIPTION = "Посчитать количество слов и символов в тексте."
PARAMS = {"text": "текст для анализа"}


def run(ctx, text: str = "", **kwargs) -> str:
    words = len(text.split())
    chars = len(text)
    lines = len(text.splitlines()) or (1 if text else 0)
    return f"слов: {words}, символов: {chars}, строк: {lines}"
