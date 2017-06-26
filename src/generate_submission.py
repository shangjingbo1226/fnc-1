from utils import *
from main import train_and_predict_3_steps
import codecs
import csv

TRAIN_BODY_CSV = 'data/train_bodies.csv'
TRAIN_STANCE_CSV = 'data/train_stances.csv'

TEST_BODY_CSV = 'data/test_bodies.csv'
TEST_HEADLINE_CSV = 'data/test_stances_unlabeled.csv'

print TEST_BODY_CSV, TEST_HEADLINE_CSV

id2body, id2body_sentences = load_body(TRAIN_BODY_CSV)
test_id2body, test_id2body_sentences = load_body(TEST_BODY_CSV)

for (body_id, body) in test_id2body.iteritems():
    if body_id in id2body and body != id2body[body_id]:
        print '[Fatal Error] body_id is ambiguous!'
        exit(-1)

id2body.update(test_id2body)
id2body_sentences.update(test_id2body_sentences)

train_data = load_stance(TRAIN_STANCE_CSV)[1:]

seen_head = set()
seen_body_id = set()
for (head, body_id, stance) in train_data:
    seen_head.add(' '.join(head))
    seen_body_id.add(body_id)

test_data = load_title(TEST_HEADLINE_CSV)
print len(test_data)

overlap = 0
for (head, body_id) in test_data:
    if ' '.join(head) in seen_head or body_id in seen_body_id:
        overlap += 1
print 'overlap =', float(overlap) / len(test_data)

test_pred, test_scores = train_and_predict_3_steps(train_data, test_data, id2body, id2body_sentences)
print len(test_pred), len(test_scores)

with open('results/submission_scores.csv', 'w') as out:
    for scores in test_scores:
        out.write(','.join([str(score) for score in scores]) + '\n')

def load_dataset(filename):
    with open(filename) as fh:
        reader = csv.DictReader(fh)
        data = list(reader)
    return data

test_dataset = load_dataset(TEST_HEADLINE_CSV)

with open('results/submission.csv', 'w') as csvfile:
    fieldnames = ['Headline', 'Body ID', 'Stance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, t in enumerate(test_dataset):
        t['Stance'] = test_pred[i]
        writer.writerow(t)

