import numpy as np
import quapy as qp
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Ordinal.model import RegressionQuantification, LogisticAT, LogisticSE, LogisticIT, LAD, OrdinalRidge
from quapy.method.aggregative import PACC, CC, EMQ, PCC, ACC
from os.path import join
from utils import load_samples_folder, load_single_sample_pkl
from evaluation import nmd, mnmd
from tqdm import tqdm


"""
This script generates all results from Table 1 in the paper, i.e., all results comparing quantifiers equipped with
standard logistic regression against quantifiers equipped with order-aware classifiers
"""

def quantifiers():
    params_LR = {'C': np.logspace(-3,3,7), 'class_weight': [None, 'balanced']}
    params_OLR = {'alpha':np.logspace(-3, 3, 7), 'class_weight': [None, 'balanced']}
    params_SVR = {'C': np.logspace(-3,3,7), 'class_weight': [None, 'balanced']}
    params_Ridge = {'alpha': np.logspace(-3, 3, 7), 'class_weight': [None, 'balanced'], 'normalize':[True,False]}

    # baselines
    yield 'CC(LR)', CC(LogisticRegression()), params_LR
    yield 'PCC(LR)', PCC(LogisticRegression()), params_LR
    yield 'ACC(LR)', ACC(LogisticRegression()), params_LR
    yield 'PACC(LR)', PACC(LogisticRegression()), params_LR
    yield 'SLD(LR)', EMQ(LogisticRegression()), params_LR

    # with order-aware classifiers
    # threshold-based ordinal regression (see https://pythonhosted.org/mord/)
    yield 'CC(OLR-AT)', CC(LogisticAT()), params_OLR
    yield 'PCC(OLR-AT)', PCC(LogisticAT()), params_OLR
    yield 'ACC(OLR-AT)', ACC(LogisticAT()), params_OLR
    yield 'PACC(OLR-AT)', PACC(LogisticAT()), params_OLR
    yield 'SLD(OLR-AT)', EMQ(LogisticAT()), params_OLR

    yield 'CC(OLR-SE)', CC(LogisticSE()), params_OLR
    yield 'PCC(OLR-SE)', PCC(LogisticSE()), params_OLR
    yield 'ACC(OLR-SE)', ACC(LogisticSE()), params_OLR
    yield 'PACC(OLR-SE)', PACC(LogisticSE()), params_OLR
    yield 'SLD(OLR-SE)', EMQ(LogisticSE()), params_OLR

    yield 'CC(OLR-IT)', CC(LogisticIT()), params_OLR
    yield 'PCC(OLR-IT)', PCC(LogisticIT()), params_OLR
    yield 'ACC(OLR-IT)', ACC(LogisticIT()), params_OLR
    yield 'PACC(OLR-IT)', PACC(LogisticIT()), params_OLR
    yield 'SLD(OLR-IT)', EMQ(LogisticIT()), params_OLR
    # other options include mord.LogisticIT(alpha=1.), mord.LogisticSE(alpha=1.)

    # regression-based ordinal regression (see https://pythonhosted.org/mord/) 
    yield 'CC(LAD)', CC(LAD()), params_SVR
    yield 'ACC(LAD)', ACC(LAD()), params_SVR
    yield 'CC(ORidge)', CC(OrdinalRidge()), params_Ridge
    yield 'ACC(ORidge)', ACC(OrdinalRidge()), params_Ridge


def run_experiment(params):
    qname, q, param_grid = params
    qname += posfix
    resultfile = join(resultpath, f'{qname}.all.csv')
    if os.path.exists(resultfile):
        print(f'result file {resultfile} already exists: continue')
        return None

    print(f'fitting {qname} for all-drift')


    def load_test_samples():
        folderpath = join(datapath, domain, protocol, 'test_samples')
        for sample in tqdm(load_samples_folder(folderpath, filter=None, load_fn=load_sample_fn), total=5000):
            if posfix == '-std':
                sample.instances = zscore.transform(sample.instances)
            yield sample.instances, sample.prevalence()


    def load_dev_samples():
        folderpath = join(datapath, domain, protocol, 'dev_samples')
        for sample in tqdm(load_samples_folder(folderpath, filter=None, load_fn=load_sample_fn), total=1000):
            if posfix == '-std':
                sample.instances = zscore.transform(sample.instances)
            yield sample.instances, sample.prevalence()

    q = qp.model_selection.GridSearchQ(
        q,
        param_grid,
        sample_size=1000,
        protocol='gen',
        error=mnmd,
        val_split=load_dev_samples,
        n_jobs=-1,
        refit=False,
        timeout=60*60*2,
        verbose=True).fit(train)

    hyperparams = f'{qname}\tall\t{q.best_params_}\t{q.best_score_}'

    print('[done]')

    report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd])
    mean_nmd = report['nmd'].mean()
    std_nmd = report['nmd'].std()
    print(f'{qname}: {mean_nmd:.4f} +-{std_nmd:.4f}')
    report.to_csv(resultfile, index=False)

    print('[learning regressor-based adjustment]')
    q = RegressionQuantification(q.best_model(), val_samples_generator=load_dev_samples)
    q.fit(None)

    report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd])
    mean_nmd = report['nmd'].mean()
    std_nmd = report['nmd'].std()
    print(f'[{qname} regression-correction] {mean_nmd:.4f} +-{std_nmd:.4f}')
    resultfile = join(resultpath, f'{qname}.all.reg.csv')
    report.to_csv(resultfile, index=False)

    return hyperparams


if __name__ == '__main__':
    domain = 'Books-roberta-base-finetuned-pkl/checkpoint-1188-average'
    #domain = 'Books-tfidf'
    posfix = ''

    # domain = 'fact'
    # posfix = '-std'  # set to '' to avoid standardization
    # posfix = ''

    load_sample_fn = load_single_sample_pkl
    datapath = './data'
    protocol = 'app'
    resultpath = join('./results', domain, protocol)
    os.makedirs(resultpath, exist_ok=True)

    train = load_sample_fn(join(datapath, domain), 'training_data')

    if posfix=='-std':
        zscore = StandardScaler()
        train.instances = zscore.fit_transform(train.instances)

    with open(join(resultpath, 'hyper.txt'), 'at') as foo:
        hypers = qp.util.parallel(run_experiment, quantifiers(), n_jobs=-3)
        for h in hypers:
            if h is not None:
                foo.write(h)
                foo.write('\n')


