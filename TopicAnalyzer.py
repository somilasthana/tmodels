import sys
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary, MmCorpus
import nltk
from nltk.corpus import stopwords
from PConfig import PConfig
from PConstant import PConstant

import logging
import StopWord
import re

class TopicAnalyzer(object):

    def __init__(self, textcolname):

        self._flogger()
        self.corpus = MmCorpus(PConstant.CORPUS_DIR_PATH.value + textcolname +'_corpus.mm')
        self.dictionary = Dictionary.load(PConstant.DICTIONARY_DIR_PATH.value + textcolname + '_dictionary.dict')
        self.lda = models.LdaModel.load(PConstant.LDA_DIR_PATH.value + textcolname + '_lda.model')
        self.stopwords = StopWord.EnglishStopWord().stopwords()

    def top_topics(self, num_topics):
        self._logger.info("discover top topics '%d'", num_topics)
        topiclist = {}
        ttlist = self.lda.top_topics(self.corpus, num_topics)
        for cnt,tt in enumerate(ttlist):
            topiclist.setdefault(cnt, [ t for v,t in tt ])
        return topiclist

    def get_topic_dist(self, doc):

        self._logger.info("get topic distribution")
        clz_doc = self.__cleanze(doc)
        vec_bow = self.dictionary.doc2bow(clz_doc)#self.__prep_dict(clz_doc)
        return self.lda[vec_bow]

    def __cleanze(self, doc):
        self._logger.info("cleanzing document")

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
        clz_doc = filterfunc(doc.split()) 
        return clz_doc
 
    def __prep_dict(self,doc):

        def nltk_stopwords():
            return set(nltk.corpus.stopwords.words('english'))
        additional_stopwords = ['nbsp', '.', ',', '"', "'", '?', '!','>', ':', ';', '(', ')', '[', ']', '{', '}','/', '.com']
        dictionary = Dictionary(doc)
        stopwords = nltk_stopwords().union(additional_stopwords)
        stopword_ids = map(dictionary.token2id.get, stopwords)
        dictionary.filter_tokens(stopword_ids)
        dictionary.compactify()
        return dictionary.doc2bow(doc)


    def get_tags(self, doc, threshold=0.4):

        topicdist = self.get_topic_dist(doc)
        int_topics = []

        for topic in topicdist:
            if topic[1] > threshold:
                int_topics.append(topic[0])

        tag_set = []
        tag_map = {}
        for topicid in int_topics:
            tag_map.setdefault(topicid, [])
            for word_dist in self.lda.show_topic(topicid):
                tag_map[topicid].append(word_dist[0])
                
        for topicid in tag_map:
            for word in tag_map[topicid]:
                if not word in tag_set:
                    tag_set.append(word)

        return tag_set

    def _flogger(self):

        self._logger = logging.getLogger('TopicAnalyzer')
        self._logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)
    
