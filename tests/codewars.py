import string

import inspect
import random
from typing import Optional, Any


# print(inspect.getsource(string.Formatter))
# print(dir(string))
# help(str.find("текст", "текст"))

# def to_camel_case(raw_text) -> Optional[str]:
#     if not (raw_text and raw_text.strip()): return None
#
#     trans_text = str.maketrans("-_", "  ")
#     raw_text = raw_text.translate(trans_text).split()
#
#     for i in range(1, len(raw_text)):
#         raw_text[i] = raw_text[i].capitalize()
#
#     raw_text = "".join(raw_text)
#
#     return raw_text
#
#
# text: str = " "
# text_1: str = "the_stealth_warrior"
# text_2: str = "The-Stealth-Warrior"
#
# print(f"result to_camel_case:", to_camel_case(text_1))

# def valid_braces(bracket) -> bool:
#     if not bracket or len(bracket) % 2 != 0:
#         return False
#
#     brick = {"(": ")", "{": "}", "[": "]"}
#     stack = []
#
#     for char in bracket:
#         if char in brick:
#             stack.append(char)
#         elif char in brick.values():
#             if not stack or brick[stack.pop()] != char:   # }
#                 return False
#     return len(stack) == 0

# brackets: str = "(){}[]"
# brackets: str = "[({})](]"
# print(f"result valid_braces:", valid_braces(brackets))

# def two_sum(numbers, target) -> list[int] | list[Any]:
#     seen: dict[int, int] = {}
#
#     for i, num in enumerate(numbers):
#         diff = target - num
#
#         if diff in seen:
#             return (seen[diff], i)
#
#         seen[num] = i
#
#     return []
#
# print(two_sum([1, 2, 3], 4)) # returns (0, 2) or (2, 0)
# print(two_sum([3, 2, 4], 6)) # returns (1, 2) or (2, 1)

# from preloaded import MORSE_CODE
#
# def decode_morse(morse_code):
#     morse_code = morse_code.strip()
#
#     if not morse_code:
#         return ''
#
#     words = morse_code.split('   ')
#
#     decoded_words = []
#     for word in words:
#         if not word:
#             continue
#
#         characters = word.split(' ')
#
#         decoded_chars = []
#         for char in characters:
#             if not char:
#                 continue
#
#             decoded_char = MORSE_CODE[char]
#             decoded_chars.append(decoded_char)
#
#         decoded_word = ''.join(decoded_chars)
#         decoded_words.append(decoded_word)
#
#     return ' '.join(decoded_words)

# print(decode_morse('....   . -.--'))  # Должно вернуть "H EY", а не "HEY"








































