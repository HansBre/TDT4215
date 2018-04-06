from datetime import datetime
import argparse
from src.db import FulltextDb


def process_article(data):
    published = find_published(data)

    word_count = find_word_count(data)

    if published is None and word_count is None:
        return None

    return {
        'published': published,
        'word_count': word_count,
    }


def find_published(data):
    try:
        raw_published = data['publishtime']
    except KeyError:
        return None
    return datetime.strptime(raw_published, '%Y-%m-%dT%H:%M:%S.%fZ')


def find_word_count(data):
    try:
        body_lines = data['body']
    except KeyError:
        return None
    body = ' '.join(body_lines)
    words = body.split()
    return len(words)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze the fulltext data set and update the DB with the '
                    'information gathered.',
        add_help=False,
    )
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Re-evaluate all articles, even those with a '
                             'value in the database from before.')
    FulltextDb.populate_argparser(parser)
    parser.add_argument('tarfile', help='The contentdata.tar.gz file.')
    args = parser.parse_args()

    db = FulltextDb.create_from_args(args)
    try:
        db.update(
            args.tarfile,
            ['published', 'word_count'],
            process_article,
            overwrite=args.overwrite,
        )
    finally:
        db.close()


if __name__ == '__main__':
    main()
