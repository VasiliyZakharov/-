from collections import defaultdict
from prettytable import PrettyTable
import mysql.connector
from mysql.connector import Error
from poetry_analyzer import PoetryAnalyzer, process_text


def print_stats(genre_stats, year_stats):
    print("\nСтатистика по жанрам:")
    for genre, stats in genre_stats.items():
        table = PrettyTable()
        table.title = genre
        table.field_names = ["Категория", "Топ значений", "Доля (%)", "Примеры"]

        counters = {
            'meter': defaultdict(int),
            'scheme': defaultdict(int),
            'rhyme': defaultdict(int),
            'alliteration': defaultdict(int),
            'assonance': defaultdict(int),
            'inexact_total': 0,
            'inexact_inexact': 0,
            'rhyme_richness_sum': 0.0,
            'rhyme_richness_count': 0
        }

        total = defaultdict(int)

        for entry in stats:
            for cat in ['meter', 'scheme', 'rhyme', 'alliteration', 'assonance']:
                if cat in entry:
                    if cat == 'scheme':
                        scheme = entry[cat]
                        counters[cat][scheme] += 1
                        total[cat] += 1
                    else:
                        for key, val in entry[cat].items():
                            counters[cat][key] += val
                            total[cat] += val
            counters['inexact_total'] += entry.get('inexact_rhymes', {}).get('total_pairs', 0)
            counters['inexact_inexact'] += entry.get('inexact_rhymes', {}).get('inexact_pairs', 0)
            counters['rhyme_richness_sum'] += entry.get('rhyme_richness', 0)
            counters['rhyme_richness_count'] += 1

        for category in ['meter', 'scheme', 'rhyme', 'alliteration', 'assonance']:
            sorted_items = sorted(counters[category].items(), key=lambda x: -x[1])[:3]
            top_values = [f"{k} ({v})" for k, v in sorted_items]
            examples = ', '.join(list(counters[category].keys())[:3]) if counters[category] else '-'
            percentages = [f"{(v / total[category] * 100):.1f}%" for k, v in sorted_items] if total[category] > 0 else [
                '0%']

            table.add_row([
                category.capitalize(),
                '\n'.join(top_values),
                ', '.join(percentages),
                examples
            ])

        if counters['inexact_total'] > 0:
            inexact_percent = (counters['inexact_inexact'] / counters['inexact_total'] * 100)
        else:
            inexact_percent = 0

        avg_richness = (counters['rhyme_richness_sum'] / counters['rhyme_richness_count']) if counters[
                                                                                                  'rhyme_richness_count'] > 0 else 0

        table.add_row(
            ['Неточные рифмы', f"{counters['inexact_inexact']}/{counters['inexact_total']}", f"{inexact_percent:.1f}%",
             ''])
        table.add_row(['Богатство рифм', f"{avg_richness:.2f}", '', ''])

        print(table)

    print("\nАнализ по годам выпуска:")
    year_table = PrettyTable()
    year_table.field_names = [
        "Год", "Песен", "Топ размеров (%)", "Топ схем (%)",
        "Топ рифм (%)", "Аллитер./песню", "Ассонанс/песню",
        "Неточные (%)", "Богатство"
    ]

    for year in sorted(year_stats.keys(), reverse=True):
        data = year_stats[year]
        total_songs = data['songs']
        format_percent = lambda d: ', '.join(
            [f"{k} ({(v / total_songs * 100):.1f}%)" for k, v in d.items()][:2]) if total_songs > 0 else '-'

        inexact_percent = (data['inexact_inexact'] / data['inexact_total'] * 100) if data['inexact_total'] > 0 else 0
        avg_richness = (data['rhyme_richness_sum'] / data['songs']) if data['songs'] > 0 else 0

        year_table.add_row([
            year,
            total_songs,
            format_percent(data['meter']),
            format_percent(data['scheme']),
            format_percent(data['rhyme']),
            f"{data['alliteration'] / total_songs:.1f}" if total_songs > 0 else '-',
            f"{data['assonance'] / total_songs:.1f}" if total_songs > 0 else '-',
            f"{inexact_percent:.1f}%",
            f"{avg_richness:.2f}"
        ])

    print(year_table)

    genre_rating_table = PrettyTable()
    genre_rating_table.title = "Оценка жанров"
    genre_rating_table.field_names = ["Жанр", "Оценка", "Компоненты оценки"]
    genre_rating_data = {}

    for genre, entries in genre_stats.items():
        total_entries = len(entries)

        total_inexact = sum(e['inexact_rhymes']['inexact_pairs'] for e in entries)
        total_pairs = sum(e['inexact_rhymes']['total_pairs'] for e in entries)
        inexact_percent = (total_inexact / total_pairs * 100) if total_pairs > 0 else 0

        richness_avg = sum(e['rhyme_richness'] for e in entries) / total_entries

        has_allit_asson = any(len(e['alliteration']) > 0 or len(e['assonance']) > 0 for e in entries)

        complex_scheme = any(
            len(e['scheme']) > 4 or
            not re.match(r'^(AABB|ABAB|ABBA|AAAA)$', e['scheme'])
            for e in entries
        )

        has_non_male = any(
            any(rt != 'мужская' for rt in e['rhyme'].keys())
            for e in entries
        )

        score = (inexact_percent * 0.30 +
                 richness_avg * 100 * 0.30 +
                 0.1 * has_allit_asson +
                 0.15 * complex_scheme +
                 0.15 * has_non_male)

        components = [
            f"Неточные: {inexact_percent:.1f}%",
            f"Богатство: {richness_avg:.2f}",
            "Аллит/Ассон: Да" if has_allit_asson else "Нет",
            "Сложная схема" if complex_scheme else "Простая схема",
            "Разные рифмы" if has_non_male else "Только мужские"
        ]

        genre_rating_data[genre] = (score, components)

    for genre, (score, components) in sorted(genre_rating_data.items(), key=lambda x: -x[1][0]):
        genre_rating_table.add_row([genre, f"{score:.2f}", "\n".join(components)])

    print("\nОценка жанров:")
    print(genre_rating_table)

    year_rating_table = PrettyTable()
    year_rating_table.title = "Оценка по годам"
    year_rating_table.field_names = ["Год", "Оценка", "Компоненты оценки"]
    year_rating_data = {}

    for year, data in year_stats.items():
        total_songs = data['songs']
        if total_songs == 0:
            continue

        inexact_percent = (data['inexact_inexact'] / data['inexact_total'] * 100) if data['inexact_total'] > 0 else 0
        richness_avg = data['rhyme_richness_sum'] / total_songs if total_songs > 0 else 0

        has_allit_asson = (data['alliteration'] + data['assonance']) > 0

        complex_scheme = any(
            len(entry['scheme']) > 4 or
            not re.match(r'^(AABB|ABAB|ABBA|AAAA|AAAAAA|AAAAAAAA|AAAAAAAB)$', entry['scheme'])
            for e in genre_stats.values()
            for entry in e
            if str(year) in entry.get('years', [])
        )

        has_non_male = any(
            rt != 'мужская'
            for e in genre_stats.values()
            for entry in e
            if str(year) in entry.get('years', [])
            for rt in entry['rhyme'].keys()
        )

        score = (inexact_percent * 0.30 +
                 richness_avg * 100 * 0.30 +
                 0.1 * has_allit_asson +
                 0.15 * complex_scheme +
                 0.15 * has_non_male)

        components = [
            f"Неточные: {inexact_percent:.1f}%",
            f"Богатство: {richness_avg:.2f}",
            "Аллит/Ассон: Да" if has_allit_asson else "Нет",
            "Сложная схема" if complex_scheme else "Простая схема",
            "Разные рифмы" if has_non_male else "Только мужские"
        ]

        year_rating_data[year] = (score, components)

    for year, (score, components) in sorted(year_rating_data.items(), key=lambda x: -x[1][0]):
        year_rating_table.add_row([year, f"{score:.2f}", "\n".join(components)])

    print("\nОценка по годам:")
    print(year_rating_table)


def main():
    connection = None
    cursor = None
    analyzer = PoetryAnalyzer()
    genre_stats = defaultdict(list)
    year_stats = defaultdict(lambda: {
        'songs': 0,
        'meter': defaultdict(int),
        'scheme': defaultdict(int),
        'rhyme': defaultdict(int),
        'alliteration': 0,
        'assonance': 0,
        'inexact_total': 0,
        'inexact_inexact': 0,
        'rhyme_richness_sum': 0.0,
        'rhyme_richness_count': 0
    })

    try:
        connection = mysql.connector.connect(
            host="185.114.245.170",
            port=3306,
            database="cz63101_songs",
            user="cz63101_songs",
            password="88005553535dfcz2003_"
        )

        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("""
                SELECT G.Name, P.Name, S.Name, T.Fragment, YEAR(S.Release)
                FROM Genres AS G
                JOIN Performers AS P ON G.Id = P.IdGenre
                JOIN Songs AS S ON P.Id = S.IdPerformer
                JOIN Texts AS T ON S.Id = T.IdSong
                ORDER BY G.Name, P.Name, S.Name
            """)

            col_widths = {
                'genre': 15, 'performer': 20, 'song': 25, 'year': 6,
                'meter': 12, 'scheme': 8, 'rhyme': 18,
                'allit': 18, 'asson': 18, 'inexact': 12
            }

            header = f"{'Жанр':<{col_widths['genre']}} " \
                     f"{'Исполнитель':<{col_widths['performer']}} " \
                     f"{'Песня':<{col_widths['song']}} " \
                     f"{'Год':<{col_widths['year']}} " \
                     f"{'Размер':<{col_widths['meter']}} " \
                     f"{'Схема':<{col_widths['scheme']}} " \
                     f"{'Типы рифм':<{col_widths['rhyme']}} " \
                     f"{'Аллитерации':<{col_widths['allit']}} " \
                     f"{'Ассонансы':<{col_widths['asson']}} " \
                     f"{'Неточные (%)':<{col_widths['inexact']}}"

            print(header)
            print("-" * (sum(col_widths.values()) + 5))

            for record in cursor.fetchall():
                genre, performer, song, text, release_year = record
                translit_text = process_text(text)
                analysis = analyzer.full_analysis(translit_text)
                rhyme_info = analyzer.analyze_rhyme_types(translit_text)

                genre_entry = {
                    'meter': {analysis['meter']: 1},
                    'scheme': analysis['rhyme_scheme'],
                    'rhyme': rhyme_info['counts'],
                    'alliteration': dict(analysis['alliteration']),
                    'assonance': dict(analysis['assonance']),
                    'inexact_rhymes': analysis['inexact_rhymes'],
                    'rhyme_richness': analysis['rhyme_richness'],
                    'years': [str(release_year)] if release_year else []
                }
                genre_stats[genre].append(genre_entry)

                if release_year:
                    year = str(release_year)
                    year_data = year_stats[year]
                    year_data['songs'] += 1
                    year_data['meter'][analysis['meter']] += 1
                    year_data['scheme'][analysis['rhyme_scheme']] += 1
                    year_data['rhyme_richness_sum'] += analysis['rhyme_richness']
                    year_data['rhyme_richness_count'] += 1

                    for rhyme_type, count in rhyme_info['counts'].items():
                        year_data['rhyme'][rhyme_type] += count

                    year_data['alliteration'] += sum(c for _, c in analysis['alliteration'])
                    year_data['assonance'] += sum(c for _, c in analysis['assonance'])
                    year_data['inexact_total'] += analysis['inexact_rhymes']['total_pairs']
                    year_data['inexact_inexact'] += analysis['inexact_rhymes']['inexact_pairs']

                rhyme_str = ", ".join([f"{k}:{v}" for k, v in rhyme_info['counts'].items()])
                allit_str = ", ".join([f"{k}({v})" for k, v in analysis['alliteration']])
                asson_str = ", ".join([f"{k}({v})" for k, v in analysis['assonance']])
                inexact_percent = analysis['inexact_rhymes']['percentage']

                output = f"{genre:<{col_widths['genre']}} " \
                         f"{performer:<{col_widths['performer']}} " \
                         f"{song:<{col_widths['song']}} " \
                         f"{release_year or '':<{col_widths['year']}} " \
                         f"{analysis['meter']:<{col_widths['meter']}} " \
                         f"{analysis['rhyme_scheme'][:col_widths['scheme']]:<{col_widths['scheme']}} " \
                         f"{rhyme_str:<{col_widths['rhyme']}} " \
                         f"{allit_str:<{col_widths['allit']}} " \
                         f"{asson_str:<{col_widths['asson']}} " \
                         f"{inexact_percent:.1f}%"
                print(output)

    except Error as e:
        print(f"Ошибка базы данных: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

    print_stats(genre_stats, year_stats)


if __name__ == "__main__":
    main()
