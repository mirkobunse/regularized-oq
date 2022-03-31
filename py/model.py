import mord
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.miscmodels.ordinal_model import OrderedModel


class OrderedLogisticRegression:
    def __init__(self, model='logit'):
        assert model in ['logit', 'probit'], 'unknown ordered model, valid ones are logit or probit'
        self.model = model

    def fit(self, X, y):
        if issparse(X):
            self.svd = TruncatedSVD(500)
            X = self.svd.fit_transform(X)
        self.learner = OrderedModel(y, X, distr=self.model)
        self.res_prob = self.learner.fit(method='bfgs', disp=False, skip_hessian=True)

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        if issparse(X):
            assert hasattr(self, 'svd'), \
                'X matrix in predict is sparse, but the method has not been fit with sparse type'
            X = self.svd.transform(X)
        return self.res_prob.model.predict(self.res_prob.params, exog=X)


class LAD(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, class_weight=None):
        self.C = C
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        self.regressor = LinearSVR(C=self.C)
        # self.regressor = SVR()
        # self.regressor = Ridge(normalize=True)
        classes = sorted(np.unique(y))
        self.nclasses = len(classes)
        if self.class_weight == 'balanced':
            class_weight = compute_class_weight('balanced', classes=classes, y=y)
            sample_weight = class_weight[y]
        self.regressor.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        r = self.regressor.predict(X)
        c = np.round(r)
        c[c<0]=0
        c[c>(self.nclasses-1)]=self.nclasses-1
        return c.astype(np.int)

    # def predict_proba(self, X):
    #     r = self.regressor.predict(X)
    #     nC = len(self.classes_)
    #     r = np.clip(r, 0, nC - 1)
    #     dists = np.abs(np.tile(np.arange(nC), (len(r), 1)) - r.reshape(-1,1))
    #     invdist = 1 - dists
    #     invdist[invdist < 0] = 0
    #     return invdist

    def decision_function(self, X):
        r = self.regressor.predict(X)
        nC = len(self.classes_)
        dists = np.abs(np.tile(np.arange(nC), (len(r), 1)) - r.reshape(-1,1))
        invdist = 1 - dists
        return invdist

    @property
    def classes_(self):
        return np.arange(self.nclasses)

    def get_params(self, deep=True):
        return {'C':self.C, 'class_weight': self.class_weight}

    def set_params(self, **params):
        self.C = params['C']
        self.class_weight = params['class_weight']


class OrdinalRidge(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, class_weight=None, normalize=False):
        self.alpha = alpha
        self.class_weight = class_weight
        self.normalize = normalize

    def fit(self, X, y, sample_weight=None):
        self.regressor = Ridge(alpha=self.alpha, normalize=self.normalize)
        classes = sorted(np.unique(y))
        self.nclasses = len(classes)
        if self.class_weight == 'balanced':
            class_weight = compute_class_weight('balanced', classes=classes, y=y)
            sample_weight = class_weight[y]
        self.regressor.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        r = self.regressor.predict(X)
        c = np.round(r)
        c[c<0]=0
        c[c>(self.nclasses-1)]=self.nclasses-1
        return c.astype(np.int)

    # def predict_proba(self, X):
    #     r = self.regressor.predict(X)
    #     nC = len(self.classes_)
    #     r = np.clip(r, 0, nC - 1)
    #     dists = np.abs(np.tile(np.arange(nC), (len(r), 1)) - r.reshape(-1,1))
    #     invdist = 1 - dists
    #     invdist[invdist < 0] = 0
    #     return invdist

    def decision_function(self, X):
        r = self.regressor.predict(X)
        nC = len(self.classes_)
        dists = np.abs(np.tile(np.arange(nC), (len(r), 1)) - r.reshape(-1,1))
        invdist = 1 - dists
        return invdist

    @property
    def classes_(self):
        return np.arange(self.nclasses)

    def get_params(self, deep=True):
        return {'alpha':self.alpha, 'class_weight': self.class_weight, 'normalize': self.normalize}

    def set_params(self, **params):
        self.alpha = params['alpha']
        self.class_weight = params['class_weight']
        self.normalize = params['normalize']


# with order-aware classifiers
# threshold-based ordinal regression (see https://pythonhosted.org/mord/)
class LogisticAT(mord.LogisticAT):
    def __init__(self, alpha=1.0, class_weight=None):
        assert class_weight in [None, 'balanced'], 'unexpected value for class_weight'
        self.class_weight = class_weight
        super(LogisticAT, self).__init__(alpha=alpha)

    def fit(self, X, y, sample_weight=None):
        if self.class_weight == 'balanced':
            classes = sorted(np.unique(y))
            class_weight = compute_class_weight('balanced', classes=classes, y=y)
            sample_weight = class_weight[y]
        return super(LogisticAT, self).fit(X, y, sample_weight=sample_weight)


class LogisticSE(mord.LogisticSE):
    def __init__(self, alpha=1.0, class_weight=None):
        assert class_weight in [None, 'balanced'], 'unexpected value for class_weight'
        self.class_weight = class_weight
        super(LogisticSE, self).__init__(alpha=alpha)

    def fit(self, X, y, sample_weight=None):
        if self.class_weight == 'balanced':
            classes = sorted(np.unique(y))
            class_weight = compute_class_weight('balanced', classes=classes, y=y)
            sample_weight = class_weight[y]
        return super(LogisticSE, self).fit(X, y, sample_weight=sample_weight)


class LogisticIT(mord.LogisticIT):
    def __init__(self, alpha=1.0, class_weight=None):
        assert class_weight in [None, 'balanced'], 'unexpected value for class_weight'
        self.class_weight = class_weight
        super(LogisticIT, self).__init__(alpha=alpha)

    def fit(self, X, y, sample_weight=None):
        if self.class_weight == 'balanced':
            classes = sorted(np.unique(y))
            class_weight = compute_class_weight('balanced', classes=classes, y=y)
            sample_weight = class_weight[y]
        return super(LogisticIT, self).fit(X, y, sample_weight=sample_weight)


# regression-based ordinal regression (see https://pythonhosted.org/mord/)
# class LAD(mord.LAD):
#     def fit(self, X, y):
#         self.classes_ = sorted(np.unique(y))
#         return super().fit(X, y)


# class OrdinalRidge(mord.OrdinalRidge):
#     def fit(self, X, y):
#         self.classes_ = sorted(np.unique(y))
#         return super().fit(X, y)

