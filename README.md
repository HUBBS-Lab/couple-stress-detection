# couple-stress-detection

##### The PyTorch implementation of paper 'A semi-supervised few-shot learning approach with domain adaptation for personalized stress detection within dating couples'.

The data file used in the experiment 'preprocessed_data_valid_couples_v3.csv'.

Four sample running results and the related environmental settings are included in the 'code' folder:

- [oneshot_Siamese_sample_selection.ipynb](https://github.com/HUBBS-Lab-TAMU/couple-stress-detection/blob/main/code/oneshot_Siamese_sample_selection.ipynb) and [oneshot_Siamese_sample_selection_distance_mapping](https://github.com/HUBBS-Lab-TAMU/couple-stress-detection/blob/main/code/oneshot_Siamese_sample_selection_distance_mapping.ipynb) refers to the baseline and proposed method as in the paper under a one-shot learning setting (the first stressed and unstressed samples from each couple is available).
- [Siamese_sample_selection.ipynb](https://github.com/HUBBS-Lab-TAMU/couple-stress-detection/blob/main/code/Siamese_sample_selection.ipynb) and [Siamese_sample_selection_distance_mapping.ipynb](https://github.com/HUBBS-Lab-TAMU/couple-stress-detection/blob/main/code/Siamese_sample_selection_distance_mapping.ipynb) refers to the baseline and proposed method under a few-shot learning setting (the first 40% of samples from each couple is available).
