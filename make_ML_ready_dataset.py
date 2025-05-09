import pandas as pd

# load patient/sample data 
sample_data = pd.read_excel("/Users/anikaflorin/Documents/Thesis/data/sample_data.xlsx")
sample_data["Sample_ID"] = sample_data["Sample_ID"].astype(str)
# load raw transcript counts (for each sample)
raw_counts= pd.read_csv("/Users/anikaflorin/Documents/Thesis/data/GSE243375_All_Sample_Raw_Counts.csv",index_col=0)
raw_counts_transposed = raw_counts.T
# merge sample and transcript data
merged_df = sample_data.merge(raw_counts_transposed,left_on="Sample_ID",right_index=True,how="left")
merged_df.to_csv('fullfeatures.csv', index=False)
merged_df = merged_df.drop(columns=['Sample_ID','Patient_ID','HER2_heterogeneity','RCB_categories'])

merged_df.to_csv('ML-ready-unfiltered.csv', index=False)
# remove columns with only zeros
filtered_merged_df = merged_df.loc[:, (merged_df != 0).any() & merged_df.notna().any()]
filtered_merged_df.to_csv('ML-ready-filtered.csv', index=False)


# # seperate dataset into two based on HER2 hetergoeniety status
# nohet_subset = filtered_merged_df[filtered_merged_df.iloc[:,3]==0]
# nohet_subset.to_csv('nohet_subset.csv', index=False)

# het_subset = filtered_merged_df[filtered_merged_df.iloc[:,3]==1]
# het_subset.to_csv('het_subset.csv', index=False)