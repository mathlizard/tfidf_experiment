"""
1. Compute the most important key-words (a key-word can be between 1-3 words) 
2. Choose the top n words from the previously generated list. Compare these keywords 
with all the words occurring in all of the transcripts. 
3. Generate a score (rank) for these top n words based on analysed transcripts. 
~
Call it with 
'test()'
or
'test(n)'
or
'first_task()'
or
'second_task(n)'
or
'third_task(n)'
where n should be an integer, respectively.
"""
import nltk
import string
import math
import heapq
import codecs
import sys


def tf(word, freq_dict):
    """ returns the term frequency of word, where freq_dict is the 
        dictionary of word:no_of_occurence pairs """
    return freq_dict[word]


def no_of_documents_containing_word(word, documentlist):
    """ returns the number of documents in documentlist containing 
        word """
    return sum(1 if word in document else 0 for document in documentlist)


def idf(word, documentlist):
    """ returns the inverse document frequency of word with respect to 
        documentlist """
    return math.log(len(documentlist) / (1 + no_of_documents_containing_word(word, documentlist)))


def tfidf(word, freq_dict, documentlist):
    """ returns the tf-idf value of word with respect to freq_dict and 
        documentlist """
    return tf(word, freq_dict) * idf(word, documentlist)


def get_words(filename):
    """ returns the list of words in filename, all lowercase, 
        no punctuation """
    with codecs.open(filename, 'rb', 'utf-8') as file:
        tokens = nltk.tokenize.word_tokenize(file.read())
        return [word.lower() for word in tokens if word not in string.punctuation]


def get_words_lists(files):
    """ returns a list of lists, containing the words in files, one list 
        per file, respectively """
    return [get_words(file) for file in files]


def importance(unigram_modifier=1.0, bigram_modifier=1.2, trigram_modifier=1.4, tr_list=None):
    """ returns a dictionary of keyword:importance values, where keyword 
        is either a unigram, bigram or a trigram """
    if tr_list is None:
        tr_list = ["transcript_1.txt", "transcript_2.txt", "transcript_3.txt"]
    # default values are set
    script = get_words("script.txt")
    uni_gram_transcripts = get_words_lists(tr_list)
    freq_unigram_script = dict(nltk.FreqDist(script))
    importance_unigrams = {word: unigram_modifier * tfidf(word, freq_unigram_script, uni_gram_transcripts) 
                           for word in script}
    # 
    bi_gram_script = list(nltk.bigrams(script))
    freq_bigram_script = dict(nltk.FreqDist(bi_gram_script))
    bi_gram_transcripts = [nltk.bigrams(transcript) for transcript in uni_gram_transcripts]
    importance_bigrams = {word: bigram_modifier * tfidf(word, freq_bigram_script, bi_gram_transcripts) 
                          for word in bi_gram_script}
    # 
    tri_gram_script = list(nltk.trigrams(script))
    freq_trigram_script = dict(nltk.FreqDist(tri_gram_script))
    tri_gram_transcripts = [nltk.trigrams(transcript) for transcript in uni_gram_transcripts]
    importance_trigrams = {word: trigram_modifier * tfidf(word, freq_trigram_script, tri_gram_transcripts) 
                           for word in tri_gram_script}
    # 
    return {**importance_unigrams, **importance_bigrams, **importance_trigrams}


def get_top_n_important_keywords(n, importance_dict=None):
    """ returns the top n most important keywords """
    if importance_dict is None:
        importance_dict = importance()
    # default values are set
    return heapq.nlargest(n, importance_dict, key=importance_dict.get)


def get_top_n_important_keywords_notuple(n, importance_dict=None):
    """ return the top n most important keywords, but not as tuples """
    top_n_keywords = get_top_n_important_keywords(n, importance_dict)
    # transforming tuples of str to str, while leaving alone str
    return list(map(lambda word: word if isinstance(word,str) else " ".join(word), top_n_keywords))


def get_words_occuring_in_all_transcripts(tr_list=None):
    """ returns the list of words which occur in each transcript """
    if tr_list is None:
        tr_list = ["transcript_1.txt", "transcript_2.txt", "transcript_3.txt"]
    # default values are set
    uni_gram_transcripts = get_words_lists(tr_list)
    filterfunc = lambda x: x in uni_gram_transcripts[1] and x in uni_gram_transcripts[2]
    return list(filter(filterfunc, set(uni_gram_transcripts[0])))


def first_task():
    """ 1. Compute the most important key-words (a key-word can be 
        between 1-3 words) """
    return importance()


def second_task(n, importance_dict=None, top_n_keywords=None):
    """ 2. Choose the top n words from the previously generated list. 
        Compare these keywords with all the words occurring in all of 
        the transcripts. """
    if importance_dict is None:
        importance_dict = first_task()
    if top_n_keywords is None:
        top_n_keywords = get_top_n_important_keywords_notuple(n, importance_dict)
    # default values are set
    words_occuring_in_all_transcripts = get_words_occuring_in_all_transcripts()
    # now let's 'compare' them
    line = "top " + str(n) + " keywords"
    print(line + "\n" + "-"*len(line))
    print(top_n_keywords)
    print()
    print("the words occuring in all transcripts:")
    print("--------------------------------------")
    print(words_occuring_in_all_transcripts)
    print()


def third_task(n, importance_dict=None, top_n_keywords=None, tr_list=None):
    """ 3. Generate a score (rank) for these top n words based on 
        analysed transcripts. """
    if importance_dict is None:
        importance_dict = first_task()
    if top_n_keywords is None:
        top_n_keywords = get_top_n_important_keywords_notuple(n, importance_dict)
    if tr_list is None:
        tr_list = ["transcript_1.txt", "transcript_2.txt", "transcript_3.txt"]
    # default values are set
    uni_gram_transcripts = get_words_lists(tr_list)
    freq_unigram_transcripts = [dict(nltk.FreqDist(transcript)) for transcript in uni_gram_transcripts]
    importance_transcripts = [0]*len(uni_gram_transcripts)
    for i in range(len(importance_transcripts)):
        importance_transcripts[i] = {word: tfidf(word, freq_unigram_transcripts[i], uni_gram_transcripts[i]) 
                                     for word in uni_gram_transcripts[i]}
    # now let's compute the score = sum of importances wrt transcripts
    score = {}
    for i, transcript in enumerate(importance_transcripts):
        for word in top_n_keywords:
            score[(word,"transcript_"+str(1+i)+".txt")] = sum(transcript[t] 
                                                              for t in word.split())
    # the task is to generate it, not to print it to the console
    return score


def test(n=20):
    """ if you want to test all the tasks ~ call me """
#       should write a proper unittest, subclassing TestCase ~ 
#       but I was a lazy bastard 
    print("FIRST TASK: ")
    importance_dict = first_task()
    print(importance_dict)
    print()
    print("SECOND TASK WITH n="+str(n)+": ")
    top_n_keywords = get_top_n_important_keywords_notuple(n, importance_dict)
    second_task(n, importance_dict, top_n_keywords)
    print("THIRD TASK WITH n="+str(n)+": ")
    score = third_task(n, importance_dict, top_n_keywords)
    for line in score.items():
        print("score( '"+line[0][0]+"' in",line[0][1],") = ",line[1])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        test()
    elif sys.argv[1].isdigit():
        i = int(sys.argv[1])
        if i == 1:
            print("FIRST TASK: ")
            importance_dict = first_task()
            print(importance_dict)
            print()
        elif i == 2:
            n = 20
            print("SECOND TASK WITH n="+str(n)+": ")
            importance_dict = first_task()
            top_n_keywords = get_top_n_important_keywords_notuple(n, importance_dict)
            second_task(n, importance_dict, top_n_keywords)
        elif i == 3:
            n = 20
            print("THIRD TASK WITH n="+str(n)+": ")
            importance_dict = first_task()
            top_n_keywords = get_top_n_important_keywords_notuple(n, importance_dict)
            score = third_task(n, importance_dict, top_n_keywords)
            for line in score.items():
                print("score( '"+line[0][0]+"' in",line[0][1],") = ",line[1])
        else:
            print("invalid argument ~ must be a number between 1 and 3")
    else:
        print("invalid argument ~ must be a number between 1 and 3")
        
    
# installer: 
# --------- 
# import nltk
# nltk.download('punkt')

