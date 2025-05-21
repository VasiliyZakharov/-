import nltk
import re
from nltk.corpus import cmudict
from collections import defaultdict, Counter
from ruaccent import RUAccent
import spacy
from spacy.symbols import VERB, NOUN, ADJ

# Инициализация ресурсов NLP
try:
    nltk.download('cmudict', quiet=True)
    nltk.download('punkt', quiet=True)
    nlp = spacy.load("ru_core_news_sm")
except Exception as e:
    print(f"Ошибка загрузки ресурсов: {e}")
    exit(1)


class PoetryAnalyzer:
    def __init__(self):
        self.accent = RUAccent()
        self.accent.load(omograph_model_size='turbo', use_dictionary=True)
        self.nlp = nlp
        self.vowels = {'а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я'}
        self.stress_symbols = {'´', '+', '\u0301'}
        self.consonants = set('бвгджзклмнпрстфхцчшщ')

    def _clean_word(self, word):
        word = re.sub(r'\W+$', '', word)
        word = re.sub(r'\d', '', word)
        return word.lower().replace("ё", "е")

    def clean_line(self, line):
        return re.sub(r'[^\w\s-]', '', line).strip()

    def _get_contextual_stress(self, word, pos, context):
        try:
            doc = self.nlp(context)
            for token in doc:
                if token.text.lower() == word.lower():
                    if pos == VERB:
                        if token.tag_ == 'VERB':
                            if token.lemma_.endswith(('ться', 'тись')):
                                return len(word) - 4
                            if word.endswith(('ли', 'ло', 'ла', 'лся')):
                                return len(word) - 3
                    elif pos == NOUN:
                        if token.morph.get('Case') == ['Nom']:
                            if word.endswith(('ая', 'яя')):
                                return len(word) - 2
                            if word.endswith('ость'):
                                return len(word) - 4
                    elif pos == ADJ:
                        if word.endswith(('ый', 'ий', 'ой')):
                            return len(word) - 2
                        if word.endswith(('ая', 'яя')):
                            return len(word) - 2
            return None
        except Exception as e:
            print(f"Ошибка контекстного анализа: {e}")
            return None

    def get_stress_pattern(self, stressed_line):
        pattern = []
        for word in stressed_line.split():
            stress_pos = max((word.find(sym) for sym in self.stress_symbols), default=-1)
            if stress_pos == -1:
                continue
            clean_word = re.sub(r'[\'´+\u0301]', '', word.lower())
            vowels = re.findall(r'[аеёиоуыэюя]', clean_word)
            if not vowels:
                continue
            stress_syllable = len(re.findall(r'[аеёиоуыэюя]', clean_word[:stress_pos])) + 1
            pattern.append(stress_syllable)
        return pattern

    def detect_meter(self, pattern):
        if len(pattern) < 2:
            return 'Недостаточно данных'
        diffs = [pattern[i] - pattern[i - 1] for i in range(1, len(pattern))]
        freq = defaultdict(int)
        for d in diffs:
            if d > 0:
                freq[d] += 1
        if not freq:
            return 'Неопределен'
        common = max(freq, key=freq.get)
        meter_map = {1: 'Хорей', 2: 'Ямб', 3: 'Дактиль', 4: 'Амфибрахий', 5: 'Анапест'}
        return meter_map.get(common, 'Смешанный')

    def _get_rhyme_type(self, word, context=None):
        word = self._clean_word(word)
        if not word or len(word) < 3:
            return None

        doc = self.nlp(word)
        pos = doc[0].pos if doc else None
        contextual_stress = None

        if context:
            contextual_stress = self._get_contextual_stress(word, pos, context)

        try:
            stressed_word = self.accent.process_word(word)
            if '´' not in stressed_word:
                stressed_word = self.accent.apply_stress_paradigm(word)
        except:
            stressed_word = word

        vowels_in_word = []
        stressed_index = -1
        for i, char in enumerate(stressed_word.lower()):
            if char in self.vowels:
                vowels_in_word.append(i)
                if '´' in stressed_word[i]:
                    stressed_index = len(vowels_in_word) - 1

        if stressed_index == -1:
            if contextual_stress is not None:
                stressed_index = contextual_stress
            elif word.endswith(('ла', 'ло', 'ли', 'лся')):
                stressed_index = len(vowels_in_word) - 2
            elif word.endswith(('ая', 'яя', 'ь')):
                stressed_index = len(vowels_in_word) - 2
            else:
                stressed_index = len(vowels_in_word) - 1

        position_from_end = len(vowels_in_word) - stressed_index - 1

        if position_from_end == 0:
            return 'male'
        elif position_from_end == 1:
            return 'female'
        elif position_from_end == 2:
            return 'dactylic'
        elif position_from_end >= 3:
            return 'hyperdactylic'
        return 'unknown'

    def _get_rhyme_key(self, word):
        word = self._clean_word(word)
        for i, char in enumerate(word):
            if char in self.stress_symbols and i > 0:
                if word[i - 1].lower() in self.vowels:
                    return word[i - 1:].lower()
        return word[-3:].lower() if len(word) >= 3 else word.lower()

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def analyze_rhyme_scheme(self, text, threshold=2):
        poem_lines = [self.clean_line(line) for line in text.split('\n') if line.strip()]
        processed_lines = [self.accent.process_all(line) for line in poem_lines]

        rhyme_keys = []
        for line in processed_lines:
            words = line.split()
            last_word = words[-1] if words else ''
            rhyme_keys.append(self._get_rhyme_key(last_word))

        scheme = []
        unique = []
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for key in rhyme_keys:
            found = False
            for i, (exist_key, letter) in enumerate(unique):
                dist = self._levenshtein_distance(key, exist_key)
                if dist <= threshold:
                    scheme.append(letter)
                    found = True
                    break
            if not found:
                new_letter = letters[len(unique)] if len(unique) < len(letters) else 'Z'
                unique.append((key, new_letter))
                scheme.append(new_letter)
        return {
            'scheme': ''.join(scheme),
            'rhyme_keys': rhyme_keys,
            'groups': unique
        }

    def count_inexact_rhymes(self, rhyme_scheme_result):
        scheme = rhyme_scheme_result['scheme']
        rhyme_keys = rhyme_scheme_result['rhyme_keys']
        groups = defaultdict(list)
        for idx, letter in enumerate(scheme):
            groups[letter].append((idx, rhyme_keys[idx]))

        total_pairs = 0
        inexact_pairs = 0

        for letter, items in groups.items():
            n = len(items)
            if n < 2:
                continue
            total_pairs += n * (n - 1) // 2
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i][1] != items[j][1]:
                        inexact_pairs += 1
        return total_pairs, inexact_pairs

    def calculate_rhyme_richness(self, rhyme_scheme_result):
        scheme = rhyme_scheme_result['scheme']
        unique_rhymes = len(set(scheme))
        total_lines = len(scheme)
        return unique_rhymes / total_lines if total_lines > 0 else 0

    def analyze_rhyme_types(self, text):
        poem_lines = [self.clean_line(line) for line in text.split('\n') if line.strip()]
        processed_lines = [self.accent.process_all(line) for line in poem_lines]

        rhyme_counter = defaultdict(int)
        total_rhymes = 0

        for line, raw_line in zip(processed_lines, poem_lines):
            words = line.split()
            if not words:
                continue

            last_word = words[-1]
            rhyme_type = self._get_rhyme_type(last_word, context=raw_line)
            if rhyme_type:
                rhyme_counter[rhyme_type] += 1
                total_rhymes += 1

        type_names = {
            'male': 'мужская',
            'female': 'женская',
            'dactylic': 'дактилическая',
            'hyperdactylic': 'гипердактилическая',
            'unknown': 'неизвестно'
        }

        return {
            'counts': {type_names[k]: v for k, v in rhyme_counter.items()},
            'total': total_rhymes
        }

    def analyze_poem(self, text):
        lines = [self.clean_line(line) for line in text.split('\n') if line.strip()]
        results = []
        for line in lines:
            try:
                stressed_line = self.accent.process_all(line)
                stress_pattern = self.get_stress_pattern(stressed_line)
                results.append({
                    'stressed_line': stressed_line,
                    'pattern': stress_pattern,
                    'meter': self.detect_meter(stress_pattern)
                })
            except Exception as e:
                results.append({'meter': 'Неопределен', 'stressed_line': line})
        return results

    def analyze_alliteration(self, text, threshold=3):
        lines = [self.clean_line(line) for line in text.split('\n') if line.strip()]
        alliterations = []

        for line in lines:
            words = line.split()
            starters = []
            for word in words:
                if not word:
                    continue
                first_char = transliterate_word(word[0]).lower()
                if first_char in self.consonants:
                    starters.append(first_char)

            counter = Counter(starters)
            for char, count in counter.items():
                if count >= threshold:
                    alliterations.append((char, count))

        return sorted(alliterations, key=lambda x: (-x[1], x[0]))[:5]

    def analyze_assonance(self, text, threshold=3):
        lines = [self.clean_line(line) for line in text.split('\n') if line.strip()]
        assonances = []

        for line in lines:
            translit_line = process_text(line).lower()
            vowels = re.findall(r'[аеёиоуыэюя]', translit_line)

            counter = Counter(vowels)
            for vowel, count in counter.items():
                if count >= threshold:
                    assonances.append((vowel, count))

        return sorted(assonances, key=lambda x: (-x[1], x[0]))[:5]

    def full_analysis(self, text):
        poem_analysis = self.analyze_poem(text)
        meters = [line['meter'] for line in poem_analysis]
        main_meter = max(set(meters), key=meters.count) if meters else 'Неопределен'

        rhyme_scheme_result = self.analyze_rhyme_scheme(text)
        total_pairs, inexact_pairs = self.count_inexact_rhymes(rhyme_scheme_result)
        rhyme_richness = self.calculate_rhyme_richness(rhyme_scheme_result)

        return {
            'rhyme_scheme': rhyme_scheme_result['scheme'],
            'meter': main_meter,
            'rhyme_type': self.analyze_rhyme_types(text),
            'alliteration': self.analyze_alliteration(text),
            'assonance': self.analyze_assonance(text),
            'detailed_analysis': poem_analysis,
            'inexact_rhymes': {
                'total_pairs': total_pairs,
                'inexact_pairs': inexact_pairs,
                'percentage': (inexact_pairs / total_pairs * 100) if total_pairs > 0 else 0
            },
            'rhyme_richness': rhyme_richness
        }


def create_translit_mapping():
    return [
        ('shch', 'щ'), ('sch', 'щ'), ('ch', 'ч'), ('zh', 'ж'),
        ('kh', 'х'), ('ts', 'ц'), ('yu', 'ю'), ('ya', 'я'),
        ('yo', 'ё'), ('sh', 'ш'), ('je', 'ж'), ('ii', 'ый'),
        ('iy', 'ий'), ('a', 'а'), ('b', 'б'), ('v', 'в'),
        ('g', 'г'), ('d', 'д'), ('e', 'е'), ('z', 'з'),
        ('i', 'и'), ('y', 'й'), ('k', 'к'), ('l', 'л'),
        ('m', 'м'), ('n', 'н'), ('o', 'о'), ('p', 'п'),
        ('r', 'р'), ('s', 'с'), ('t', 'т'), ('u', 'у'),
        ('f', 'ф'), ('h', 'х'), ('c', 'ц'), ('j', 'дж'),
        ('q', 'к'), ('x', 'кс'), ('w', 'в'), ("'", 'ь'),
        ('"', 'ъ'),
    ]


def transliterate_word(word: str) -> str:
    mapping = create_translit_mapping()
    mapping.sort(key=lambda x: (-len(x[0]), x[0]))
    result = []
    pos = 0
    while pos < len(word):
        match = None
        for key, value in mapping:
            if word.lower().startswith(key, pos):
                match = (key, value)
                break
        if match:
            result.append(match[1])
            pos += len(match[0])
        else:
            result.append(word[pos])
            pos += 1
    return ''.join(result)


def process_text(text: str) -> str:
    parts = re.findall(r'([a-zA-Z]+)|([^a-zA-Z]+)', text)
    result = []
    for eng_part, other_part in parts:
        if eng_part:
            words = nltk.word_tokenize(eng_part)
            for word in words:
                result.append(transliterate_word(word))
        else:
            result.append(other_part)
    return ''.join(result)
