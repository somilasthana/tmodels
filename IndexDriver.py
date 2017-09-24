import pandas as pd
import nltk
from gensim import corpora, models, similarities
import gensim
from nltk.corpus import stopwords
import sys
reload(sys)
import numpy as np
from gensim.corpora import Dictionary, MmCorpus
import re
import string
sys.setdefaultencoding('utf-8')
import logging
import StopWord

from PConfig import PConfig
from PConstant import PConstant


class IndexDriver(object):

    def __init__(self, datarecord, colnnames, delimtr):
        self._flogger()
        self.datarecord = datarecord
        self.df = pd.read_csv(self.datarecord, names=colnnames,  delimiter=delimtr)
        self.stopwords = StopWord.EnglishStopWord().stopwords()
       

    def __cleanze(self, textcolname):
        self._logger.info("tokenzing = '%s' ", textcolname)
        self.df["tokenized"] = self.df[textcolname].astype(unicode).apply(nltk.word_tokenize)

        def filterfunc(x):
            fx = []
            for item in x:
                if item not in self.stopwords:
                    if len(item) > 3:
                        if not item.isdigit():
                            if not re.search('[0-9]', item):
                                if item.isalpha():
                                    fx.append(item.lower())
            return fx
        self._logger.info("filtering = '%s' ", textcolname)
        self.df['tokens'] = self.df['tokenized'].apply(filterfunc)

    def __corpus(self):
    
        def nltk_stopwords():
            return set(nltk.corpus.stopwords.words('english'))

        def prep_tfidf_corpus(docs, additional_stopwords=set(), no_below=2, no_above=0.05):

            dictionary = Dictionary(docs)
            stopwords = nltk_stopwords().union(additional_stopwords)
            stopword_ids = map(dictionary.token2id.get, stopwords)
            dictionary.filter_tokens(stopword_ids)
            dictionary.compactify()
            dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
            dictionary.compactify()
            corpus = [dictionary.doc2bow(doc) for doc in docs]
            self.tfidf = models.TfidfModel(corpus)
            corpus_tfidf = self.tfidf[corpus]
            return dictionary, corpus_tfidf

        dictionary, corpus = prep_tfidf_corpus(self.df['tokens'], ['nbsp', '.', ',', '"', "'", '?', '!','>', ':', ';', '(', ')', '[', ']', '{', '}','/', '.com'])
        return (dictionary, corpus)

    def __model(self):

        lsi = gensim.models.lsimodel.LsiModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics)  
        return lsi

    def indexer(self, textcolname, num_topics):
        self.num_topics = num_topics
  
        self._logger.info("cleanzing data for '%s'", textcolname)
        self.__cleanze(textcolname)
        self._logger.info("creating corpus and dictionary for '%s'", textcolname)
        self.dictionary, self.corpus = self.__corpus()
        self._logger.info("applying lda model '%s'", textcolname)
        self.lsi = self.__model()
        self._logger.info("saving models for '%s'", textcolname)
        MmCorpus.serialize(PConstant.CORPUS_DIR_PATH.value + textcolname +'_tfidf_corpus.mm', self.corpus)
        self.dictionary.save( PConstant.DICTIONARY_DIR_PATH.value + textcolname + '_tfidf_dictionary.dict')
        self.lsi.save( PConstant.LDA_DIR_PATH.value + textcolname + '_lsi.model')

    def _flogger(self):

        self._logger = logging.getLogger('IndexDriver')
        self._logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

