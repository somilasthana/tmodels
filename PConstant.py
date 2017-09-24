from enum import Enum
class PConstant(Enum) : 
    ELASTICSEARCHIP_CONFIG = "index.elasticsearch.connect"
    ES_TRIALDATA_INDEX = "trial_partica_articles"
    ES_TRIALDATA_TYPE = "testarticles" 
    TRIALDATA_SCHEMA = 0 # 28th Aug 
    CONSTANT_ARTICLEID = "articleid"

    CORENLP_CONFIG = "com.corenlp.path"

    ENTITY = "entityname"
    TAGS = "tags"

    CORPUS_DIR_PATH = "/tmp/"
    DICTIONARY_DIR_PATH = "/tmp/"
    LDA_DIR_PATH = "/tmp/"
