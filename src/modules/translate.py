'''
Used for back-translation.

Pre-requisite:
pip install googletrans==3.1.0a0
'''

from googletrans import Translator

translator = Translator()

def to_not_english(text, target, translator=translator):
    translated = translator.translate(text=text, dest=target)
    # print(translated.text)
    # print(f"Translated to {target}.")
    return translated

def to_english(text, target='en', translator=translator):
    translated = translator.translate(text=text, dest=target)
    # print(f"Translated to {target}.")
    return translated

def back_translation(text, target):
    '''
    text: can be a string or a list of strings
    '''
    not_english = to_not_english(text, target)
    translated = []
    if type(not_english) == list:
        for i in not_english:
            translated.append(to_english(i.text))
    else:
        translated = to_english(not_english.text)
    return translated
