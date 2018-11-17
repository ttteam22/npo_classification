from typing import List, Dict
import re
import itertools
import string
import logging
import collections
import numpy as np
import pandas as pd
from utils import LazyProperty

# NLP
import nltk
from nltk.corpus import stopwords
import pymorphy2
from pymystem3 import Mystem
from natasha import NamesExtractor
from alphabet_detector import AlphabetDetector
from razdel import tokenize

import word_lists
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s %(funcName)s %(message)s')

# Custom types
Tokenlist = List[str]


class PreprocessingInterface(object):
    def __init__(self):
        # Pre-loading objects
        self.mystem = Mystem()
        self.names_extractor = NamesExtractor()
        self.pymorphy = pymorphy2.MorphAnalyzer()
        self.alphabet_detector = AlphabetDetector()
        self.fallback_counter = 0

        # Dicts
        # self.en_dict = enchant.DictWithPWL("en_US", self.proj_path + '/Preprocessing/Dicts/IT_EN_dict.txt')
        # self.ru_aot_dict = enchant.Dict("ru_RU")
        self.stop_words = set(word_lists.yandex_seo_stopwords +
                              stopwords.words('russian'))
        self.unwanted_punct = ",.:!?0#№«»()-\"'_="
        self.unwanted_trans = str.maketrans(self.unwanted_punct,
                                            ''.join([' ' for x in self.unwanted_punct]))
        self.padding_punct = """!"#$%&\'()*+,;<=>?[\\]^`{|}~/«»"""
        self.full_punct = string.punctuation + '«-–»'


    def split_paragraph(raw: str):
        normalized_string = PreprocessingInterface.normalize(raw)  # replacing ("\r", "\t", ".") with "\n"
        split_strings = re.split('\n', normalized_string)
        filtering = list(filter(lambda x: x if x != " " else None, split_strings))
        filtered = [x.strip() for x in filtering]
        return filtered

    # ======================================== #
    # ########## STRING PROCESSING ########### #
    # ======================================== #
    
    def makeshift_clean(self, txt):
        txt = txt.replace('\xa0', ' ')
        txt_list = txt.split('\n')
        txt = ' '.join([x.lower() for x in txt_list])
        txt = txt.translate(self.unwanted_trans)
        return ' '.join(txt.lower().split())
    
    @staticmethod
    def normalize(raw: str) -> str:
        """ 1. lower
            2. strip
            3. remove line break symbols
        """
        raw = raw.replace('\xa0', ' ')
        line_break_cleaning = str.maketrans('\r\t.', '\n\n\n')
        result = raw.lower().translate(line_break_cleaning).strip()
        return result

    def pad_punctuation(self, raw: str, punct_list=None) -> str:
        """
        Adds whitespaces before and after each punctuation symbol
        Used to control tokenization
        """
        normal_text = raw.strip()
        padding_punctuation = punct_list if punct_list else self.padding_punct
        for char in padding_punctuation:
            normal_text = normal_text.replace(char, ' ' + char + ' ')
        return normal_text

    @staticmethod
    def nltk_tokenize(raw: str) -> Tokenlist:
        return nltk.word_tokenize(raw)

    @staticmethod
    def razdel_tokenize(raw: str):
        return list(tokenize(raw))

    def is_punct(self, token) -> bool:
        """ True only if all chars are punct """
        for c in token:
            if c not in self.full_punct:
                return False
        return True

    def remove_punct(self, text: Tokenlist) -> Tokenlist:
        return [token for token in text if not self.is_punct(token)]

    @staticmethod
    def remove_digits(tokens: list):
        return [t for t in tokens if not t.isdigit()]

    @staticmethod
    def contains_digits(token: str) -> bool:
        return any(char.isdigit() for char in token)

    def contains_punct(self, token: str) -> bool:
        return any(self.is_punct(char) for char in token)

    def is_cyrillic(self, token) -> bool:
        """
        Checks if string has only cyrillic letters
        """
        if self.contains_digits(token) or self.contains_punct(token):
            return False
        else:
            return self.alphabet_detector.only_alphabet_chars(token, 'CYRILLIC')

    # ======================================== #
    # ########### POS/LEMMATIZING ############ #
    # ======================================== #
    def lemmatize_with_mystem(self, raw: str):
        lemmatized_tokens = self.mystem.lemmatize(raw)
        lemmas_filtered = [t for t in lemmatized_tokens if t != ' ' and t != '\n']  # filter empty
        if len(lemmas_filtered) == 0:
            return ""
        return " ".join(lemmas_filtered).strip()

    def get_pymorphy_lemma(self, token: str) -> str:
        return self.pymorphy.parse(token)[0].normal_form

    def lemmatize_tokens_with_mystem(self, text: Tokenlist) -> Tokenlist:
        lemmatized_tokens = self.mystem.lemmatize(" ".join(text))
        lemmas_filtered = [t for t in lemmatized_tokens if t != ' ' and t != '\n']  # filter empty
        return lemmas_filtered

    def lemmatize_with_pymorphy(self, text: Tokenlist) -> Tokenlist:
        lemmas = []
        for token in text:
            p = self.pymorphy.parse(token)[0]
            lemmas.append(p.normal_form)
        lemmas_filtered = [t for t in lemmas if t != ' ' and t != '\n']  # filter empty
        return lemmas_filtered

    def get_mystem_pos(self, token):  # TODO: apply mystem to whole text
        response = self.mystem.analyze(token)
        analysis = response[0].get('analysis')
        try:
            the_one = analysis[0]
            tag = the_one.get('gr')
            return tag
        except Exception as e:
            print(e, e.args)
            return None

    # ======================================== #
    # ######### Mail froms cleaning ########## #
    # ======================================== #

    def parse_mystem_tags(self, analysis):
        if analysis.get("analysis"):
            if "gr" in analysis["analysis"][0]:
                tag_string = analysis["analysis"][0]["gr"]
                result = tag_string.split(",")
                return result
        return ""

    # ======================================== #
    # ############## Filtering ############### #
    # ======================================== #

    def get_vocab(self, tokenized_texts: pd.Series) -> set:
        return set(self.series_to_chain(tokenized_texts))

    def remove_stopwords(self, text: Tokenlist, stopwords: list = None) -> Tokenlist:
        if not stopwords:
            stopwords = self.stop_words
        return [t for t in text if t not in stopwords]

    @staticmethod
    def filter_by_token_length(text: Tokenlist, min=1, max=25) -> Tokenlist:
        return [t for t in text if len(t) >= min and len(t) < max]

    """
    def mystem_remove_names(self, text: Tokenlist):
        result = []
        for each in self.mystem.analyze(" ".join(text)):
            if not each['text'] in (" ", "\n"):
                if 'имя' not in tags and 'фам' not in parse_mystem_tags(each)
                    result.append(each["text"])
        return result
    """

    def pymorphy_isname(self, token: str):
        """ Better then mystem? """
        tags = self.pymorphy.parse(token)[0].tag
        if 'Name' in tags or 'Surn' in tags or 'Patr' in tags:
            return True
        else:
            return False

    def pymorphy_remove_names(self, text: Tokenlist):
        """ Takes pymorphy_isname/ as input"""
        return [t for t in text if not self.pymorphy_isname(t)]

    def get_names_df(self, df_col, name_extractor):
        ctr = collections.Counter(list(self.series_to_chain(df_col)))
        fdist_list = ctr.most_common()
        res = {each[0]: each[1] for each in fdist_list if name_extractor(each[0])}
        df = pd.DataFrame.from_dict(res, orient='index')
        df.columns = ["count"]
        df["token"] = df.index
        df.index = [list(range(len(df)))]
        return df

    # ======================================== #
    # ########### Pandas analysis ############ #
    # ======================================== #

    def get_nltk_pos_df(self, texts: pd.Series) -> pd.DataFrame:
        all_tokens = self.series_to_chain(texts)
        nltk_tags_tuple = nltk.pos_tag(all_tokens, lang='rus')
        tags = set([each[1] for each in nltk_tags_tuple])

        def get_tokens_by_tag(tag):
            # Set of tokens by input tag
            token_tag_list = list(filter(lambda x: x[1] == tag, nltk_tags_tuple))  # [token, tag]
            return [each[0] for each in token_tag_list]  # [token]

        tag_dict = collections.OrderedDict(zip(tags, [get_tokens_by_tag(tag) for tag in tags]))
        return pd.DataFrame.from_dict(tag_dict, orient='index').transpose()

    # ======================================== #
    # ########## Jupyter analysis ############ #
    # ======================================== #
    @staticmethod
    def stats_for_untokenized(texts: pd.Series):
        """ Counts symbols in series of texts """
        return sum([len(each) for each in texts])

    @staticmethod
    def series_to_chain(texts: pd.Series) -> Tokenlist:
        """ Chained tokens in Series """
        return list(itertools.chain.from_iterable(list(texts.values)))

    def stats_for_series(self, texts: pd.Series) -> pd.DataFrame:
        """DF from Series stats"""
        empty_texts_indexes = list(texts[texts.astype(str) == '[]'].index)
        empty_texts = len(empty_texts_indexes)
        token_chain = self.series_to_chain(texts)

        result = pd.DataFrame(data=[
            [len(token_chain),
             len(list(set(token_chain))),
             len(texts),
             empty_texts,
             token_chain.count('')]
        ],
            index=['Count'],
            columns=['Total tokens',
                     'Unique tokens',
                     'Total texts',
                     'Empty texts',
                     'Empty tokens'])
        return result

    @staticmethod
    def check_empty_texts(texts: pd.Series, original_df=None):
        """
        Get unprocessed text for '[]' in Series
        :returns list of indexes or pd.Index
        """
        empty_texts_indexes = list(texts[texts.astype(str) == '[]'].index)
        if original_df:
            return original_df.loc[empty_texts_indexes]
        else:
            return empty_texts_indexes

    @staticmethod
    def drop_empty_text_rows(data: pd.DataFrame, column_name: str) -> pd.Series:
        no_na = data[column_name].dropna()
        # drop_indexes = no_na[no_na.astype(str) == '[]'].index
        drop_indexes = no_na[no_na.astype(str) == '[]'].index
        return data.drop(drop_indexes)

    @staticmethod
    def plot_occurrences(data: pd.Series, expression):
        """
        Detects first occurrence of str expression in text.
        Plots index distribution of occurrences.
        """
        indexes = [text.index(expression) for text in data if expression in text]
        fig, ax = plt.subplots()
        ax.hist(indexes, range(0, 50))
        ax.set_xticks(np.arange(0, 51, 1))
        ax.set_xlabel('Position')
        ax.set_ylabel('Count')
        plt.title("Occurrence distribution")
        print(len(indexes), ' occurrences found')
        return ax

    def get_token_counts_df(self, texts: pd.Series, topn=30) -> pd.DataFrame:
        ctr = collections.Counter(list(self.series_to_chain(texts)))
        fdist_list = ctr.most_common(topn)
        tokens = [k for k, v in fdist_list]
        counts = [v for k, v in fdist_list]
        return pd.DataFrame({"token": tokens, "count": counts})

    def plot_token_frequencies(self, texts: pd.Series, topn=30) -> sns.barplot():
        """ Plot frequency distribution over corpus for top_n tokens tokens """
        get_token_counts_df = self.get_token_counts_df(texts, topn)
        sns.barplot(x="count", y="token", data=get_token_counts_df).set_xlabel('Token appearence')

    def plot_token_distribution(self, texts: pd.Series):
        """ Overall tokens lenghts distribution for series """
        token_lenghts = [len(x) for x in self.series_to_chain(texts)]
        bow_lenghts = [len(x) for x in texts]

        # Unique lens
        fig, ax = plt.subplots(ncols=2)

        ax[0].hist(token_lenghts, bins=range(0, 25))
        ax[0].set_xticks(np.arange(0, 26, 1))
        ax[0].set_xlabel('Token length')
        ax[0].set_ylabel('Count')

        ax[1].hist(bow_lenghts, bins=range(0, 25))
        ax[1].set_xticks(np.arange(0, 26, 1))
        ax[1].set_xlabel('Tokens in text')
        ax[1].set_ylabel('Count')
        return ax

    @staticmethod
    def get_most_common(data: pd.DataFrame) -> pd.DataFrame:
        # df = self.get_categories_df(series)
        result = dict()
        for col in data.columns:
            try:
                col_most_freq = data[col].value_counts().reset_index()
                tokens = col_most_freq['index']
                freqs = col_most_freq[col]
                result[col] = [(t, f) for t, f in zip(tokens, freqs)]
            except:
                result[col] = [None]
        return pd.DataFrame.from_dict(result, orient='index').transpose()

    # ======================================== #
    # ################ OTHER ################# #
    # ======================================== #
    def separate_by_category(self, texts: pd.Series) -> Dict:
        """
        Separates tokens by types of chars in it (punctuation, numbers, ...)
        :param texts: series of tokenized texts
        :return: dict of {category:[tokenlist]}
        """
        vocab = self.series_to_chain(texts)

        result = {'num_punct': [],
                  'alpha_num': [],
                  'alpha_punct': [],
                  'punct_tokens': [],
                  'numeric_tokens': [],
                  'alpha_tokens': [],
                  'alpha_num_punct': []}

        for token in vocab:
            # Add flag by symbol category
            punct = [1 for symbol in token if (symbol in self.full_punct)]
            numerics = [1 for symbol in token if (symbol.isnumeric())]
            alpha = [1 for symbol in token if (symbol.isalpha())]

            # If token contains all types
            if (punct and numerics) and alpha:
                result['alpha_num_punct'].append(token)

            # Double
            elif numerics and punct:
                result['num_punct'].append(token)

            elif numerics and alpha:
                result['alpha_num'].append(token)

            elif alpha and punct:
                result['alpha_punct'].append(token)

            # Simple
            elif punct:
                result['punct_tokens'].append(token)

            elif numerics:
                result['numeric_tokens'].append(token)

            elif alpha:
                result['alpha_tokens'].append(token)

        return result

    def get_categories_df(self, texts: pd.Series) -> pd.DataFrame:
        # make df from separation dict
        separated_categories_dict = self.separate_by_category(texts)
        categories = pd.DataFrame.from_dict(separated_categories_dict, orient='index')
        return categories.transpose()

    # ======================================== #
    # ############## PIPELINES ############### #
    # ======================================== #
    def apply_pipeline(self, raw: str) -> Tokenlist:
        """ Apply all the methods to raw string """
        normalized = self.normalize(raw)
        padded = self.pad_punctuation(normalized)
        tokenized = self.nltk_tokenize(padded)
        no_punct = self.remove_punct(tokenized)
        no_stops = self.remove_stopwords(no_punct)
        cut_by_len = [t for t in no_stops if len(t) < 25]
        lemmatized = self.lemmatize_tokens_with_mystem(cut_by_len)
        return lemmatized

    def apply_short_pipeline(self, raw: str) -> Tokenlist:
        """ Preprocessing for manual input in window form on client-side """
        normalized = self.normalize(raw)
        tokenized = self.nltk_tokenize(normalized)
        no_punct = self.remove_punct(tokenized)
        no_stopwords = self.remove_stopwords(no_punct)
        lemmatized = self.lemmatize_with_pymorphy(no_stopwords)
        return lemmatized
