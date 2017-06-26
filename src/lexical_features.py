from utils import *
import nltk
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import nltk.data
from scipy import spatial

# Load Google's pre-trained Word2Vec model.
word_vectors = KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True)

def sentences2vectors(sentences):
    body_vectors = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        cnt = 0
        body_vector = np.array([0.0] * 300)
        for word in words:
            if word in word_vectors:
                body_vector = body_vector + word_vectors[word]
                cnt += 1
        if cnt > 0:
            body_vector /= cnt
            body_vector /= np.linalg.norm(body_vector)
        body_vectors.append(body_vector)
    return body_vectors
    
def title2vectors(title):
    return sentences2vectors(tokenizer.tokenize(title))
    
def extract_features(title, body):
    title_vectors = sentences2vectors([title])
    title_vector = title_vectors[0]
    
    body_vectors = sentences2vectors(body)
    
    max_sim = -1
    diff_vector = title_vector
    for body_vector in body_vectors:
        similarity = 1 - spatial.distance.cosine(title_vector, body_vector)
        if similarity > max_sim:
            max_sim = similarity
            diff_vector = title_vector - body_vector
    features = [max_sim]
    for v in diff_vector:
        features.append(v)
    return features

id2body = load_body('data/train_bodies.csv')
train_data = load_stance('data/train_stances.csv');

out = open('train-feats.csv', 'w')
out.write('label')
for d in xrange(300):
    out.write(',dim_' + str(d))
out.write(',max_sim\n')
for (title, id, stance) in train_data:
    body = id2body[id]

    feats = extract_features(title, body)
    
    output = []
    for feat in feats:
        output.append(str(feat))
    
    out.write(stance + ',' + ','.join(output) + '\n')
out.close()