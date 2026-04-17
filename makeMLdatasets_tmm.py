import pandas as pd
import os

path = '/Users/anikaflorin/Documents/Thesis/data/'

# Load data
meta_data = pd.read_excel(os.path.join(path, 'meta_data.xlsx'))
tmm_counts = pd.read_csv(os.path.join(path, 'GSE243375_All_Samples_TMM_log2_CPM.csv'), index_col=0)

# Define subsets
core_tmm_counts = tmm_counts.loc[:, tmm_counts.columns.str.contains('core')]
res_tmm_counts = tmm_counts.loc[:, tmm_counts.columns.str.contains('Residual')]

core_meta = meta_data[meta_data['Sample_ID'].str.contains('core')]
res_meta = meta_data[meta_data['Sample_ID'].str.contains('Residual')]

meta_columns = ['Sample_ID', 'Patient_ID', 'HER2_heterogeneity', 'pCR']

# -----------------------
# Helper function
# -----------------------
def prepare_ml_df(meta, counts, outfile):
    # transpose counts
    counts_T = counts.T

    # merge
    merged = meta.merge(counts_T, left_on='Sample_ID', right_index=True, how='left')

    # extract pCR before dropping metadata
    pcr = merged['pCR']

    # expression only
    expr = merged.drop(columns=meta_columns)

    # zero filtering (genes only)
    expr_filtered = expr.loc[:, (expr != 0).any(axis=0) & expr.notna().any(axis=0)]

    # add pCR back
    final_df = pd.concat([pcr, expr_filtered], axis=1)

    # save
    final_df.to_csv(f'{path}{outfile}', index=False)

    return merged, final_df


# -----------------------
# Core + Residual full sets
# -----------------------
core_tmm_merged, ml_core = prepare_ml_df(core_meta, core_tmm_counts, 'core_tmm_ML_ready.csv')
res_tmm_merged, ml_res = prepare_ml_df(res_meta, res_tmm_counts, 'res_tmm_ML_ready.csv')


# -----------------------
# HER2 subsets (CORE)
# -----------------------
def subset_het(merged, het_value, outfile):
    subset = merged[merged['HER2_heterogeneity'] == het_value]

    pcr = subset['pCR']
    expr = subset.drop(columns=meta_columns)

    expr_filtered = expr.loc[:, (expr != 0).any(axis=0) & expr.notna().any(axis=0)]

    final_df = pd.concat([pcr, expr_filtered], axis=1)
    final_df.to_csv(f'{path}{outfile}', index=False)


subset_het(core_tmm_merged, 0, 'nohet_core_tmm_ML_ready.csv')
subset_het(core_tmm_merged, 1, 'het_core_tmm_ML_ready.csv')


# -----------------------
# HER2 subsets (RESIDUAL)
# -----------------------
subset_het(res_tmm_merged, 0, 'nohet_res_tmm_ML_ready.csv')
subset_het(res_tmm_merged, 1, 'het_res_tmm_ML_ready.csv')