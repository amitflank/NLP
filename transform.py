from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer

class ItemSelector(BaseEstimator, TransformerMixin):
        def __init__(self, key):
            self.key = key

        def fit(self, x, y=None):
            return self

        def transform(self, data_dict):
            return data_dict[self.key]

class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return [{'pos':  row['polarity'], 'sub': row['subjectivity'],  'len': row['len']} for _, row in data.iterrows()]
