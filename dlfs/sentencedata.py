import csv
import numpy as np
import itertools
import nltk
# nltk.download('punkt')


def getSentenceData(csv_file, vocab_size=8000):
    unknown_token = "UNKNOWN_TOKEN"
    start_token = "SENTENCE_START"
    end_token = "SENTENCE_END"

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = [f"{start_token} {x} {end_token}" for x in sentences]
    
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    tokenized_sentences = list(filter(lambda x: len(x) > 3, tokenized_sentences))

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))
    common_words_freq = word_freq.most_common(vocab_size-1)
    words = [x[0] for x in common_words_freq]
    words.append(unknown_token)
    vocab = dict([(w, i) for i, w in enumerate(words)])
    
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in vocab else unknown_token for w in sent]
    
    X_train = np.asarray([[vocab[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[vocab[w] for w in sent[1:]] for sent in tokenized_sentences])
    
    return vocab, words, X_train, Y_train

            


if __name__ == "__main__":
    getSentenceData("data/reddit-comments-2015-08.csv")