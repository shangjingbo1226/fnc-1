import csv
import nltk.data
import re
from sklearn import feature_extraction
import random
import os
from collections import defaultdict

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def print_confusion_matrix(cm):
    lines = ['CONFUSION MATRIX:']
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0.0
    total = 0
    related_hit = 0.0
    stance_hit = 0
    stance_total = 0.0
    for i, row in enumerate(cm):
        if i < 3:
            related_hit += sum(row[:-1])
            stance_total += sum(row)
            stance_hit += row[i]
        else:
            related_hit += row[i]
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    lines.append("ACCURACY: {:.3f}".format(hit / total))
    lines.append("ACCURACY-relatedness: {:.3f}".format(related_hit / total))
    lines.append("ACCURACY-stance: {:.3f}".format(stance_hit / stance_total))
    print('\n'.join(lines))


_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in remove_stopwords(nltk.word_tokenize(s))]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def load_body(filename):
    id2body = {}
    id2body_sentences = {}
    with open(filename) as fh:
        reader = csv.DictReader(fh)
        data = list(reader)
        for row in data:
            id = row['Body ID']
            body = unicode(row['articleBody'], errors='ignore').decode('utf-8').strip()
            body_sentences = tokenizer.tokenize(body)

            clean_body = clean(body)
            clean_body = get_tokenized_lemmas(clean_body)

            clean_body_sentences = []
            for sentence in body_sentences:
                clean_sentence = clean(sentence)
                clean_sentence = get_tokenized_lemmas(clean_sentence)
                clean_body_sentences.append(clean_sentence)
            id2body[id] = clean_body
            id2body_sentences[id] = clean_body_sentences
    return id2body, id2body_sentences


    reader = unicode_csv_reader(codecs.open(filename))
    for id, body in reader:
        body_sentences = tokenizer.tokenize(body)

        clean_body = clean(body)
        clean_body = get_tokenized_lemmas(clean_body)

        clean_body_sentences = []
        for sentence in body_sentences:
            clean_sentence = clean(sentence)
            clean_sentence = get_tokenized_lemmas(clean_sentence)
            clean_body_sentences.append(clean_sentence)
        id2body[id] = clean_body
        id2body_sentences[id] = clean_body_sentences
    return id2body, id2body_sentences

def load_title(filename):
    data = []
    with open(filename) as fh:
        reader = csv.DictReader(fh)
        raw_data = list(reader)
        for row in raw_data:
            title = unicode(row['Headline'], errors='ignore').decode('utf-8').strip()
            clean_title = clean(title)
            clean_title = get_tokenized_lemmas(clean_title)

            id = row['Body ID']
            # ignore the stance if there is any
            data.append((clean_title, id))
    return data

    reader = unicode_csv_reader(open(filename))
    for row in reader:
        title = row[0]
        clean_title = clean(title)
        clean_title = get_tokenized_lemmas(clean_title)

        id = row[1]
        # ignore the stance if there is any
        data.append((clean_title, id))
    return data
    
def load_stance(filename):
    reader = unicode_csv_reader(open(filename))
    data = []
    for title, id, stance in reader:
        clean_title = clean(title)
        clean_title = get_tokenized_lemmas(clean_title)
        data.append((clean_title, id, stance.strip()))
    return data
    