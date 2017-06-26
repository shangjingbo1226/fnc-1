# Fake News Challenge 1, 4th place's solution

## Introduction

We are the team **Chips Ahoy!** in the [Fake News Challenge](http://www.fakenewschallenge.org/). Our solution leverages both lexical matching features and semantic embedding features based on the most similar sentence. The machine learning models are gradient boosting trees implemented in the xgboost.

We have made 5 submissions and the best performed one is an aggregation between some submissions. In this repo, we release the code to generate our 1st submission (about 80.2%), which is essentially same as our best submission but relies on a single model. The winning team's score is 82.02%. If we aggregate the winning team's prediction and our 1st submission (predict as "Discuss" when two submissions disagree with each other), the score can be easily boosted to **83.00%**. 

## Result

We've put our 1st submission in the folder *submissions/* for your reference.

## Usage

```
pip install -r requirements.txt
```

```
$python nltk
Python 2.7...
...
>>> import nltk
>>> nltk.download('all')
```

We are using the Google's pre-trained word embeddings. Please following the instructions in the README in the folder *resources/* before you run the code.

```
$mkdir results
$python generate_submission.py
```

The final prediction can be found in the *results/submission.csv*.