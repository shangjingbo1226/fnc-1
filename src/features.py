from collections import defaultdict

def text2count(body):
    freq = defaultdict(int)
    for word in body:
        freq[word] += 1
    return freq

def lexical_overlaps(title, body, idf):
    features = []

    words_in_body = text2count(body)
    words_in_title = text2count(title)

    maximum, maximum_cnt = 0.0, 0.0
    for (word, cnt_title) in words_in_title.iteritems():
        maximum += cnt_title * idf[word]
        maximum_cnt += cnt_title

    overlaps, overlap_cnt = 0, 0
    for (word, cnt_title) in words_in_title.iteritems():
        if word in words_in_body:
            tf = min(cnt_title, words_in_body[word])
            overlap_cnt += tf
            overlaps += tf * idf[word]
    features += [overlaps, overlaps / maximum, overlap_cnt, overlap_cnt / maximum_cnt]

    words_in_body = text2count(body[:len(title) * 4])
    overlaps, overlap_cnt = 0, 0
    for (word, cnt_title) in words_in_title.iteritems():
        if word in words_in_body:
            tf = min(cnt_title, words_in_body[word])
            overlap_cnt += tf
            overlaps += tf * idf[word]
    features += [overlaps, overlaps / maximum, overlap_cnt, overlap_cnt / maximum_cnt]

    return features

import numpy as np

def title2vector(title, word2vec, idf):
    vector = np.array([0.0] * 300)
    cnt = 0
    for word in title:
        if word in word2vec:
            vector += word2vec[word]
            cnt += 1
    if cnt > 0:
        vector /= cnt
        vector /= np.linalg.norm(vector)
    return vector

def compute_overlap(title, body_sentence, idf):
    words_in_body = text2count(body_sentence)
    words_in_title = text2count(title)

    maximum, maximum_cnt = 0.0, 0.0
    for (word, cnt_title) in words_in_title.iteritems():
        maximum += cnt_title * idf[word]
        maximum_cnt += cnt_title

    overlaps, overlap_cnt = 0, 0
    for (word, cnt_title) in words_in_title.iteritems():
        if word in words_in_body:
            tf = min(cnt_title, words_in_body[word])
            overlap_cnt += tf
            overlaps += tf * idf[word]
    return overlaps / maximum, overlap_cnt / maximum_cnt

def semantic_similarities(title, body_sentences, word2vec, idf):
    max_overlap, max_overlap_cnt = 0, 0
    title_vector = title2vector(title, word2vec, idf)
    max_sim = -1
    best_vector = np.array([0.0] * 300)
    
    supports = []
    for sub_body in body_sentences:
        sub_body_vector = title2vector(sub_body, word2vec, idf)

        cur_overlap, cur_overlap_cnt = compute_overlap(title, sub_body, idf)

        max_overlap = max(max_overlap, cur_overlap)
        max_overlap_cnt = max(max_overlap_cnt, cur_overlap_cnt)

        similarity = 0
        for i in xrange(300):
            similarity += title_vector[i] * sub_body_vector[i]
        if similarity > max_sim:
            max_sim = similarity
            best_vector = sub_body_vector

        supports.append(similarity)

    features = [max_overlap, max_overlap_cnt, max(supports), min(supports)]

    #for v in title_vector - best_vector:
    #    features.append(v)
    for v in best_vector:
        features.append(v)
    for v in title_vector:
        features.append(v)
    return features

def extract_features(title, body, body_sentences, idf, word2vec):
    return lexical_overlaps(title, body, idf) + semantic_similarities(title, body_sentences, word2vec, idf)
