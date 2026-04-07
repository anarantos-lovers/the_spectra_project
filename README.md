 A Two-Stage Fragment-Connection Framework for Infrared-Spectrum-Driven Molecular Structure Elucidation
 
This repository contains the code and workflow used in our paper:

A Two-Stage Fragment-Connection Framework for Infrared-Spectrum-Driven Molecular Structure Elucidation

The project formulates molecular structure elucidation as a BRICS fragment-connection prediction problem. Molecules are decomposed into BRICS fragments, a Stage-2 model predicts pairwise connection types between fragment nodes, and a constrained decoder assembles ranked candidate molecules.

The main paper experiments are conducted in an IR-only setting with a pairwise MLP scorer. The repository also includes separate branches for Transformer comparison, multimodal ablation, peak-table supplementary experiments, and scaffold-level failure analysis for Supporting Information.

Main paper baseline uses the filtered (mask_sum >= 3) IR-only setting with a global vocabulary; strict-vocabulary, oracle-fragment, multimodal, transformer, and unfiltered experiments are separate controlled analyses.

 Overview

 Main paper setting

1. IR-only input  
2. BRICS fragment vocabulary  
3. Pairwise connection prediction with a Stage-2 MLP model  
4. 257 connection classes (0-255 valid BRICS type pairs, 256 = NONE)  
5. Complexity-controlled evaluation on samples with mask_sum >= 3  
6. Sparse export + constrained beam-search decoding  

 Additional experiments included in this repository

1. Transformer-based comparison under the same fragment-connection formulation  
2. Multimodal ablation experiments  
3. Peak-table supplementary experiments  
4. Scaffold-level failure analysis for Supporting Information  



 Repository contents

This repository includes code for:

1. Dataset preprocessing and scaffold annotation  
2. BRICS fragment-count (mask_sum) computation  
3. Complexity-controlled filtering with mask_sum >= 3  
4. Random and scaffold-disjoint data splitting  
5. BRICS vocabulary construction  
6. IR-only Stage-2 training with pairwise MLP scoring  
7. Sparse pairwise-logit export  
8. Constrained beam-search decoding and evaluation  
9. Strict-vocabulary experiments  
10. Oracle-fragment experiments  
11. Sparse export ablation (top16 / top32 / top64)  
12. Transformer-based comparison experiments  
13. Multimodal ablation experiments  
14. Peak-table supplementary experiments  
15. Scaffold-level failure analysis for SI  



 Data

The experiments are based on a processed dataset derived from a public multimodal spectroscopic dataset for chemistry. In this project, the processed data typically retain columns such as:

1. smiles  
2. molecular_formula  
3. ir_spectra  
4. h_nmr_peaks  
5. c_nmr_peaks  
6. hsqc_nmr_peaks  
7. msms_cfmid_fragments_negative  
8. msms_cfmid_fragments_positive  

The main paper pipeline uses IR-only input, although the repository also contains scripts for multimodal and peak-table supplementary experiments.

Example processed datasets used in the project:

1. dataset_5.8k.parquet  
2. dataset_63k.parquet  
3. complex_total.parquet (used for augmentation-related experiments)  



 Environment

Recommended environment:

1. Python 3.9–3.11  
2. PyTorch  
3. pandas  
4. numpy  
5. RDKit  
6. tqdm  
7. pyarrow  
8. matplotlib  
9. scikit-learn  

Example installation:

bash
pip install torch pandas numpy pyarrow tqdm matplotlib scikit-learn

Main workflow
The main workflow is:
1.	Prepare the dataset 
2.	Add scaffold annotations 
3.	Compute mask_sum 
4.	Filter to mask_sum >= 3 
5.	Build BRICS vocabulary 
6.	Generate random or scaffold-disjoint splits 
7.	Train the IR-only Stage-2 MLP model 
8.	Export sparse pairwise logits 
9.	Decode and evaluate 
10.	Run controlled ablations and SI analyses 

If RDKit fails to import on Linux servers with errors related to GLIBCXX or libstdc++.so.6, try the following:

1. Install runtime libraries into the conda environment:
conda install -n torch_final -c conda-forge libstdcxx-ng libgcc-ng -y

2. Prioritize the conda environment libraries:
export LD_LIBRARY_PATH=/root/miniconda3/envs/torch_final/lib:$LD_LIBRARY_PATH

3. Test RDKit separately:
python -c "from rdkit import Chem; print('RDKit ok')"

If the RDKit test passes, proceed with the preprocessing scripts such as make_scaffold_dataset.py and make_mask_sum_csv.py.

1. Preprocessing

1.1 Add scaffold annotations

python make_scaffold_dataset.py ^
  --data dataset_63k.parquet ^
  --out dataset_with_scaffold_63k.parquet

This script computes Bemis–Murcko scaffolds from smiles, removes invalid scaffold entries, and saves a scaffold-annotated parquet file.

1.2 Compute fragment-node count (mask_sum)

python make_mask_sum_csv.py ^
  --data dataset_with_scaffold_63k.parquet ^
  --out_csv mask_sum_dataset_with_scaffold_63k.csv
  
mask_sum is computed from BRICS decomposition and is used for complexity-controlled evaluation.

1.3 Filter to the complexity-controlled subset (mask_sum >= 3)

python filter_dataset_by_mask_sum_keep_scaffold.py ^
  --data dataset_with_scaffold_63k.parquet ^
  --mask_csv mask_sum_dataset_with_scaffold_63k.csv ^
  --min_mask_sum 3 ^
  --out dataset_mask3_with_scaffold_63k.parquet
This is the main filtered setting used in the paper.


2. Data splitting
  
2.1 Random split

python random_split_from_scaffold_ds.py ^
  --data dataset_mask3_with_scaffold_63k.parquet ^
  --seed 0 ^
  --out_prefix mask3_63k_random0
  
This produces files such as:
•	train_mask3_63k_random0_0.parquet 
•	val_mask3_63k_random0_0.parquet 
•	test_mask3_63k_random0_0.parquet 

2.2 Scaffold-disjoint split

python scaffold_kfold_split.py ^
  --data dataset_mask3_with_scaffold_63k.parquet ^
  --seed 0 ^
  --n_splits 10 ^
  --outdir splits_mask3_scaffold_63k
  
This produces files such as:
•	splits_mask3_scaffold_63k/train_scaffold_0.parquet 
•	splits_mask3_scaffold_63k/val_scaffold_0.parquet 
•	splits_mask3_scaffold_63k/test_scaffold_0.parquet 


3. BRICS vocabulary construction
   
3.1 Global vocabulary

python build_brics_vocab_from_train.py ^
  --train dataset_mask3_with_scaffold_63k.parquet ^
  --out_tsv vocab_global_63k.tsv ^
  --min_count 1
  
This builds a BRICS fragment vocabulary from the whole filtered dataset. The vocabulary stores fragment SMILES and counts, and the loader reserves special tokens for PAD and UNK.

3.2 Strict-vocabulary setting

python build_brics_vocab_from_train.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --out_tsv vocab_strict_random0.tsv ^
  --min_count 1
  
For the strict-vocabulary experiment, build the vocabulary only from the corresponding training split and keep that vocabulary fixed during training, export, and decoding.


4. Main model training (IR-only MLP)

The main Stage-2 model is implemented in stage2_brics_model.py. It uses:

1.	Fragment embedding 
2.	IR projection network 
3.	Pairwise MLP scorer over fragment pairs 
4.	Masked logits over invalid fragment positions 

Train the main model with:

python train_stage2_brics.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_global_63k.tsv ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --none_weight 0.0005 ^
  --out best_stage2_random0_mask3_w0005.pt
  
The trainer uses pair sampling to reduce the dominance of the NONE class in pairwise connection classification.


5. Sparse export and decoding
   
5.1 Export sparse pairwise logits
   
python export_stage2_logits_to_npz_dataset.py ^
  --ckpt best_stage2_random0_mask3_w0005.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_random0_mask3_w0005_top32.npz ^
  --vocab vocab_global_63k.tsv ^
  --batch 8 ^
  --device cuda ^
  --topk 32 ^
  --dtype float16 ^
  --cleanup_tmp
  
5.2 Decode and evaluate

python denovo_decode_eval_from_npz_fast_v1.py ^
  --npz npz_random0_mask3_w0005_top32.npz ^
  --test test_mask3_63k_random0_0.parquet ^
  --beam_size 128 ^
  --topk_out 5 ^
  --top_edge_m 4096 ^
  --top_type_r 32 ^
  --out_csv pred_random0_mask3_w0005_top32.csv
  
This performs constrained decoding and reports candidate-quality metrics such as Top-1, Top-5, validity, empty prediction rate, and Tanimoto-related reconstruction quality.

6. Main paper experiments
   
6.1 Main reconstruction experiments
   
These experiments correspond to the main paper setting:
1.	IR-only input 
2.	Random split 
3.	Scaffold-disjoint split 
4.	Complexity-controlled evaluation (mask_sum >= 3) 
5.	Pairwise MLP scorer 
6.	Sparse export + constrained decoding

   
6.2 Strict training-vocabulary experiment
  	
python train_stage2_brics.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_strict_random0.tsv ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --none_weight 0.0005 ^
  --out best_stage2_strict_random0_mask3_w0005.pt

python export_stage2_logits_to_npz_dataset.py ^
  --ckpt best_stage2_strict_random0_mask3_w0005.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_strict_random0_mask3_w0005_top32.npz ^
  --vocab vocab_strict_random0.tsv ^
  --batch 8 ^
  --device cuda ^
  --topk 32 ^
  --dtype float16 ^
  --cleanup_tmp

python denovo_decode_eval_from_npz_fast_v1.py ^
  --npz npz_strict_random0_mask3_w0005_top32.npz ^
  --test test_mask3_63k_random0_0.parquet ^
  --beam_size 128 ^
  --topk_out 5 ^
  --top_edge_m 4096 ^
  --top_type_r 32 ^
  --out_csv pred_strict_random0_mask3_w0005_top32.csv
  
This corresponds to the strict-vocabulary analysis in the paper.

6.3 Oracle-fragment experiment

python export_oracle_stage2_npz.py ^
  --ckpt best_stage2_random0_mask3_w0005.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_random0_mask3_oracle_w0005_top32.npz ^
  --vocab vocab_global_63k.tsv ^
  --batch 8 ^
  --device cuda ^
  --topk 32 ^
  --dtype float16 ^
  --cleanup_tmp

python denovo_decode_eval_from_npz_fast_v1.py ^
  --npz npz_random0_mask3_oracle_w0005_top32.npz ^
  --test test_mask3_63k_random0_0.parquet ^
  --beam_size 128 ^
  --topk_out 5 ^
  --top_edge_m 4096 ^
  --top_type_r 32 ^
  --out_csv pred_mask3_random0_w0005_oracle_top32.csv
  
This experiment supplies ground-truth BRICS fragments during evaluation and is used to distinguish fragment-coverage error from connection-prediction error.

6.4 Sparse export ablation

Run sparse export with different topk values, for example:
1.	topk = 16 
2.	topk = 32 
3.	topk = 64
   
and decode each setting separately.

6.5 Unfiltered vs filtered comparison

To reproduce the complexity-control comparison, run the same pipeline on:

•	the unfiltered scaffold-annotated dataset 
•	the filtered mask_sum >= 3 subset 

and compare reconstruction metrics.

Example (unfiltered):

python make_scaffold_dataset.py ^
  --data dataset_63k.parquet ^
  --out dataset_with_scaffold_63k.parquet

python build_brics_vocab_from_train.py ^
  --train dataset_with_scaffold_63k.parquet ^
  --out_tsv vocab_global_63k_unfiltered.tsv ^
  --min_count 1

python random_split_from_scaffold_ds.py ^
  --data dataset_with_scaffold_63k.parquet ^
  --seed 0 ^
  --out_prefix raw_63k_random0

python train_stage2_brics.py ^
  --train train_raw_63k_random0_0.parquet ^
  --val val_raw_63k_random0_0.parquet ^
  --vocab vocab_global_63k_unfiltered.tsv ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --none_weight 0.0005 ^
  --out best_stage2_63k_unfiltered_random0_w0005.pt

python export_stage2_logits_to_npz_dataset.py ^
  --ckpt best_stage2_63k_unfiltered_random0_w0005.pt ^
  --data test_raw_63k_random0_0.parquet ^
  --out npz_63k_unfiltered_random0_w0005_top32.npz ^
  --vocab vocab_global_63k_unfiltered.tsv ^
  --batch 8 ^
  --device cuda ^
  --topk 32 ^
  --dtype float16 ^
  --cleanup_tmp

python denovo_decode_eval_from_npz_fast_v1.py ^
  --npz npz_63k_unfiltered_random0_w0005_top32.npz ^
  --test test_raw_63k_random0_0.parquet ^
  --beam_size 128 ^
  --topk_out 5 ^
  --top_edge_m 4096 ^
  --top_type_r 32 ^
  --out_csv pred_63k_unfiltered_random0_w0005_top32.csv
  


7. Transformer comparison experiment
   
A separate Transformer-based comparison model is provided in:
1.	ablation_transformer_stage2_model.py 
2.	ablation_train_stage2_transformer.py 
3.	ablation_export_stage2_transformer_npz.py
   
Example:

python ablation_train_stage2_transformer.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_global_63k.tsv ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --none_weight 0.0005 ^
  --out best_transformer_random0_mask3_w0005.pt

python ablation_export_stage2_transformer_npz.py ^
  --ckpt best_transformer_random0_mask3_w0005.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_transformer_random0_mask3_w0005_top32.npz ^
  --vocab vocab_global_63k.tsv ^
  --batch 8 ^
  --device cuda ^
  --topk 32 ^
  --dtype float16 ^
  --cleanup_tmp

python denovo_decode_eval_from_npz_fast_v1.py ^
  --npz npz_transformer_random0_mask3_w0005_top32.npz ^
  --test test_mask3_63k_random0_0.parquet ^
  --beam_size 128 ^
  --topk_out 5 ^
  --top_edge_m 4096 ^
  --top_type_r 32 ^
  --out_csv pred_transformer_random0_mask3_w0005_top32.csv
  


8. Multimodal ablation experiment
   
Separate multimodal ablation scripts are provided in:
1.	ablation_stage2_multimodal_dataset.py 
2.	ablation_stage2_multimodal_model.py 
3.	ablation_train_stage2_multimodal.py 
4.	ablation_export_stage2_multimodal_npz.py 
The multimodal dataset parser supports auxiliary spectral inputs such as:
•	h_nmr_peaks 
•	c_nmr_peaks 
•	hsqc_nmr_peaks 
Example:

python ablation_train_stage2_multimodal.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_global_63k.tsv ^
  --use_h1 ^
  --use_c13 ^
  --use_hsqc ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --none_weight 0.0005 ^
  --out best_multimodal_full_random0_mask3_w0005.pt

python ablation_export_stage2_multimodal_npz.py ^
  --ckpt best_multimodal_full_random0_mask3_w0005.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_multimodal_full_random0_mask3_w0005_top32.npz ^
  --vocab vocab_global_63k.tsv ^
  --batch 8 ^
  --device cuda ^
  --topk 32 ^
  --dtype float16 ^
  --cleanup_tmp

python denovo_decode_eval_from_npz_fast_v1.py ^
  --npz npz_multimodal_full_random0_mask3_w0005_top32.npz ^
  --test test_mask3_63k_random0_0.parquet ^
  --beam_size 128 ^
  --topk_out 5 ^
  --top_edge_m 4096 ^
  --top_type_r 32 ^
  --out_csv pred_multimodal_full_random0_mask3_w0005_top32.csv
These scripts support the modality-ablation analyses referenced in the main paper.


9. Scaffold-level failure analysis for SI
    
python analyze_scaffold_failures.py ^
  --pred_csv pred_random0_mask3_w0005_top32.csv ^
  --test_parquet test_mask3_63k_random0_0.parquet ^
  --outdir scaffold_failure_random0
  
This script summarizes scaffold-level Top-1 failure counts, Top-1 failure rates, Top-5 failure rates, and typical failed cases. It is used for the scaffold-level failure analyses reported in the Supporting Information.

10. Main paper code map
Main paper
•	preprocessing and filtering
make_scaffold_dataset.py, make_mask_sum_csv.py, filter_dataset_by_mask_sum_keep_scaffold.py

•	split generation

random_split_from_scaffold_ds.py, scaffold_kfold_split.py

•	main IR-only model

stage2_brics_dataset.py, stage2_brics_model.py, train_stage2_brics.py

•	sparse export and decoding

export_stage2_logits_to_npz_dataset.py, denovo_decode_eval_from_npz_fast_v1.py 
Ablation / supplementary experiments

•	Transformer comparison

ablation_transformer_stage2_model.py, ablation_train_stage2_transformer.py, ablation_export_stage2_transformer_npz.py

•	multimodal ablation

ablation_stage2_multimodal_dataset.py, ablation_stage2_multimodal_model.py, ablation_train_stage2_multimodal.py, ablation_export_stage2_multimodal_npz.py 

•	peak-table SI experiments

si_peaktable_stage2_dataset.py, si_peaktable_stage2_model.py, si_peaktable_train.py, si_peaktable_export_npz.py 

•	scaffold failure SI

analyze_scaffold_failures.py 

11. Notes
    
The main paper claims should be reproduced with the IR-only Stage-2 MLP pipeline, not with the Transformer or multimodal branches.
Transformer and multimodal scripts are intended for controlled comparison and supplementary analysis.
Supporting Information includes scaffold-level failure portraits and supplementary peak-table analyses.

Citation
If you use this repository, please cite the paper and the processed dataset release associated with this project.

