import pandas as pd
from os.path import join
import os
from glob import glob
from pathlib import Path

from Ordinal.experiments_lr_vs_ordlr import quantifiers
from Ordinal.tabular import Table

"""
This script generates some tables for Fact-OQ (for internal use only)
"""

#domain = 'fact'
#domain = 'Books-tfidf'
domain = 'Books-roberta-base-finetuned-pkl/checkpoint-1188-average'
prot = 'app'
outpath = f'./tables/{domain}/{prot}/results.tex'

resultpath = join('./results', domain, prot)

withstd=False

methods = [qname for qname, *_ in quantifiers()]
if withstd:
    methods = [m+'-std' for m in methods]
#methods = methods + methods_variant
# methods += [m+'-r' for m in methods]

quantifiers_families = ['CC', 'PCC', 'ACC', 'PACC', 'SLD']
# method_variants = ['LR', 'OLR-AT', 'OLR-SE', 'OLR-IT', 'ORidge', 'LAD']
method_variants = ['LR', 'OLR-AT', 'OLR-IT', 'ORidge', 'LAD']
if withstd:
    method_variants = [m+'-std' for m in method_variants]

print('families:', quantifiers_families)
print('variants', method_variants)
table = Table(benchmarks=quantifiers_families, methods=method_variants, prec_mean=4, show_std=True, prec_std=4,
              color=False, show_rel_to=0, missing_str='\multicolumn{1}{c}{---}', clean_zero=True)

resultfiles = list(glob(f'{resultpath}/*).all.csv'))

for resultfile in resultfiles:
    df = pd.read_csv(resultfile)
    nmd = df['nmd'].values
    resultname = Path(resultfile).name

    method, drift, *other = resultname.replace('.csv', '').replace('-RoBERTa-average','').split('.')
    if drift!='all':
        continue
    if other:
        method += '-r'
    if method not in methods:
        continue  

    family, variant = method.split('(')
    variant = variant.replace(')', '')
    if variant not in method_variants:
        continue
    table.add(family, variant, nmd)

os.makedirs(Path(outpath).parent, exist_ok=True)

tabular = """
    \\resizebox{\\textwidth}{!}{%
            
            \\begin{tabular}{c""" + ('l' * (table.nbenchmarks)) + """} 
            \\toprule            
            """

tabular += table.latexTabularT(average=False)
tabular += """
    \end{tabular}%
    }"""

print('saving table in', outpath)
with open(outpath, 'wt') as foo:
    foo.write(tabular)
    foo.write('\n')

print('[done]')

