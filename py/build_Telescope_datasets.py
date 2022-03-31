import gzip
import quapy as qp
import numpy as np
import pandas as pd
from quapy.data import LabelledCollection
import quapy.functional as F
import os
from os.path import join
from pathlib import Path
import pickle


datadir = '../OrdinalQuantification'
outdir  = './data/'
domain = 'fact'
seed = 7

tr_size = 20000
val_size = 1000
te_size = 1000
nval = 1000
nte = 5000


def from_csv(path):
    df = pd.read_csv(path)

    # divide the continuous labels into ordered classes
    energy_boundaries = np.arange(start=2.4, stop=4.2, step=0.15)[1:-1]
    y = np.digitize(np.array(df['log10_energy'], dtype=np.float32), energy_boundaries)

    # note: omitting the dtype will result in a single instance having a different class

    # obtain a matrix of shape (n_samples, n_features)
    X = df.iloc[:, 1:].to_numpy().astype(np.float32)
    return X, y


def write_pkl(sample: LabelledCollection, path):
    os.makedirs(Path(path).parent, exist_ok=True)
    pickle.dump(sample, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)


def gen_samples_APP(pool: LabelledCollection, nsamples, sample_size, outdir, prevpath):
    os.makedirs(outdir, exist_ok=True)
    with open(prevpath, 'wt') as prevfile:
        prevfile.write('id,' + ','.join(f'{c}' for c in pool.classes_) + '\n')
        for i, prev in enumerate(F.uniform_simplex_sampling(n_classes=pool.n_classes, size=nsamples)):
            sample = pool.sampling(sample_size, *prev)
            write_pkl(sample, join(outdir, f'{i}.pkl'))
            prevfile.write(f'{i},' + ','.join(f'{p:.3f}' for p in sample.prevalence()) + '\n')


def gen_samples_NPP(pool: LabelledCollection, nsamples, sample_size, outdir, prevpath):
    os.makedirs(outdir, exist_ok=True)
    with open(prevpath, 'wt') as prevfile:
        prevfile.write('id,' + ','.join(f'{c}' for c in pool.classes_) + '\n')
        for i, sample in enumerate(pool.natural_sampling_generator(sample_size, repeats=nsamples)):
            write_pkl(sample, join(outdir, f'{i}.pkl'))
            prevfile.write(f'{i},' + ','.join(f'{p:.3f}' for p in sample.prevalence()) + '\n')



fullpath = join(datadir,domain, 'fact_wobble.csv')

data = LabelledCollection.load(fullpath, from_csv)

if np.isnan(data.instances).any():
    rows, cols = np.where(np.isnan(data.instances))
    data.instances = np.delete(data.instances, rows, axis=0)
    data.labels = np.delete(data.labels, rows, axis=0)
    print('deleted nan rows')

if np.isnan(data.instances).any():
    rows, cols = np.where(np.isnan(data.instances))
    data.instances = np.delete(data.instances, rows, axis=0)
    data.labels = np.delete(data.labels, rows, axis=0)
    print('deleted nan rows')

if np.isinf(data.instances).any():
    rows, cols = np.where(np.isinf(data.instances))
    data.instances = np.delete(data.instances, rows, axis=0)
    data.labels = np.delete(data.labels, rows, axis=0)
    print('deleted inf rows')


print(len(data))
print(data.classes_)
print(data.prevalence())

with qp.util.temp_seed(seed):
    train, rest = data.split_stratified(train_prop=tr_size)

    devel, test = rest.split_stratified(train_prop=0.5)
    print(len(train))
    print(len(devel))
    print(len(test))

    domaindir = join(outdir, domain)

    write_pkl(train, join(domaindir, 'training_data.pkl'))
    write_pkl(devel, join(domaindir, 'development_data.pkl'))
    write_pkl(test, join(domaindir, 'test_data.pkl'))

    gen_samples_APP(devel, nsamples=nval, sample_size=val_size, outdir=join(domaindir, 'app', 'dev_samples'),
                    prevpath=join(domaindir, 'app', 'dev_prevalences.txt'))
    gen_samples_APP(test, nsamples=nte, sample_size=te_size, outdir=join(domaindir, 'app', 'test_samples'),
                    prevpath=join(domaindir, 'app', 'test_prevalences.txt'))

    gen_samples_NPP(devel, nsamples=nval, sample_size=val_size, outdir=join(domaindir, 'npp', 'dev_samples'),
                    prevpath=join(domaindir, 'npp', 'dev_prevalences.txt'))
    gen_samples_NPP(test, nsamples=nte, sample_size=te_size, outdir=join(domaindir, 'npp', 'test_samples'),
                    prevpath=join(domaindir, 'npp', 'test_prevalences.txt'))



