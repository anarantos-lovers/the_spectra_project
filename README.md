 A Two-Stage Fragment-Connection Framework for Infrared-Spectrum-Driven Molecular Structure Elucidation

This repository contains the code and workflow used in our paper:

A Two-Stage Fragment-Connection Framework for Infrared-Spectrum-Driven Molecular Structure Elucidation

The project formulates molecular structure elucidation as a BRICS fragment-connection prediction problem. Molecules are decomposed into BRICS fragments, a Stage-2 model predicts pairwise connection types between fragment nodes, and a constrained decoder assembles ranked candidate molecules.

The main paper experiments are conducted in an IR-only setting with a pairwise MLP scorer. The repository also includes separate branches for Transformer comparison, multimodal ablation, peak-table supplementary experiments, and scaffold-level failure analysis for Supporting Information.



 Overview

 Main paper setting

IR-only input
BRICS fragment vocabulary
pairwise connection prediction with a Stage-2 MLP model
257 connection classes (0-255 valid BRICS type pairs, 256 = NONE)
complexity-controlled evaluation on samples with mask_sum >= 3
sparse export + constrained beam-search decoding

 Additional experiments included in this repository

Transformer-based comparison under the same fragment-connection formulation
multimodal ablation experiments
peak-table supplementary experiments
scaffold-level failure analysis for Supporting Information

 Repository contents

This repository includes code for:

dataset preprocessing and scaffold annotation
BRICS fragment-count (mask_sum) computation
complexity-controlled filtering with mask_sum >= 3
random and scaffold-disjoint data splitting
BRICS vocabulary construction
IR-only Stage-2 training with pairwise MLP scoring
sparse pairwise-logit export
constrained beam-search decoding and evaluation
strict-vocabulary experiments
oracle-fragment experiments
sparse export ablation (top16 / top32 / top64)
Transformer-based comparison experiments
multimodal ablation experiments
peak-table supplementary experiments
scaffold-level failure analysis for SI

 Data

The experiments are based on a filtered dataset derived from a public multimodal spectroscopic dataset for chemistry. In this project, the processed dataset typically retains columns such as:

smiles
molecular_formula
ir_spectra
h_nmr_peaks
c_nmr_peaks
hsqc_nmr_peaks
msms_cfmid_fragments_negative
msms_cfmid_fragments_positive

The main paper pipeline uses IR-only input, although the repository also contains scripts for multimodal and peak-table supplementary experiments.

Example processed datasets used in the project:

dataset_5.8k.parquet
dataset_63k.parquet
complex_total.parquet (used for augmentation-related experiments)



 Environment

Recommended environment:

Python 3.9–3.11
PyTorch
pandas
numpy
RDKit
tqdm
pyarrow
matplotlib
scikit-learn

Example installation:

text
pip install torch pandas numpy pyarrow tqdm matplotlib scikit-learn

Main workflow
The main workflow is:
1.	prepare the dataset 
2.	add scaffold annotations 
3.	compute mask_sum 
4.	filter to mask_sum >= 3 
5.	build BRICS vocabulary 
6.	generate random or scaffold-disjoint splits 
7.	train the IR-only Stage-2 MLP model 
8.	export sparse pairwise logits 
9.	decode and evaluate 
10.	run controlled ablations and SI analyses 

1. Preprocessing
1.1 Add scaffold annotations
python make_scaffold_dataset.py --data dataset_63k.parquet --out dataset_with_scaffold_63k.parquet
This script computes Bemis–Murcko scaffolds from smiles, removes invalid scaffold entries, and saves a scaffold-annotated parquet file.
1.2 Compute fragment-node count (mask_sum)
python make_mask_sum_csv.py --data dataset_with_scaffold_63k.parquet --out_csv mask_sum_dataset_with_scaffold_63k.csv
mask_sum is computed from BRICS decomposition and is used for complexity-controlled evaluation.
Optional distribution check:
python check_mask_sum_distribution.py --data dataset_with_scaffold_63k.parquet --out_csv mask_sum_distribution_63k.csv
This prints summary statistics and bucket counts for mask_sum.
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
train_mask3_63k_random0_0.parquet 
val_mask3_63k_random0_0.parquet 
test_mask3_63k_random0_0.parquet 
2.2 Scaffold-disjoint split
python scaffold_kfold_split.py ^
  --data dataset_mask3_with_scaffold_63k.parquet ^
  --seed 0 ^
  --n_splits 10 ^
  --outdir splits_mask3_scaffold_63k
This produces:
splits_mask3_scaffold_63k/train_scaffold_0.parquet 
splits_mask3_scaffold_63k/val_scaffold_0.parquet 
splits_mask3_scaffold_63k/test_scaffold_0.parquet 

3. BRICS vocabulary construction
3.1 Global vocabulary
python build_brics_vocab_from_train.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --out_tsv vocab_global.tsv ^
  --min_count 1
This builds a BRICS fragment vocabulary from the training file. The vocabulary stores fragment SMILES and counts, and the loader reserves special tokens for PAD and UNK.
3.2 Strict-vocabulary setting
For the strict-vocabulary experiment, build the vocabulary only from the corresponding training split, and keep that vocabulary fixed during training, export, and decoding. This matches the strict train-only vocabulary analysis described in the paper.

4. Main model training (IR-only MLP)
The main Stage-2 model is implemented in stage2_brics_model.py. It uses:
fragment embedding 
IR projection network 
pairwise MLP scorer over fragment pairs 
masked logits over invalid fragment positions 
Train the main model with:
python train_stage2_brics.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_global.tsv ^
  --ir_len 1024 ^
  --max_nodes 64 ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --device cuda ^
  --out best_stage2_random0.pt
The trainer uses pair sampling to reduce the dominance of the NONE class in pairwise connection classification.

5. Sparse export and decoding
5.1 Export sparse pairwise logits
python export_stage2_logits_to_npz_dataset.py ^
  --ckpt best_stage2_random0.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_random0_top16.npz ^
  --topk 16 ^
  --dtype float16
This exports sparse top-k pairwise logits, fragment ids, and masks for later decoding.
5.2 Decode and evaluate
python denovo_decode_eval_from_npz_fast_v1.py ^
  --npz npz_random0_top16.npz ^
  --test test_mask3_63k_random0_0.parquet ^
  --beam_size 128 ^
  --topk_out 5 ^
  --top_edge_m 4096 ^
  --top_type_r 32 ^
  --out_csv pred_random0_top16.csv
This performs constrained decoding and reports candidate-quality metrics such as Top-1, Top-5, validity, empty prediction rate, and Tanimoto-related reconstruction quality.

6. Main paper experiments
6.1 Main reconstruction experiments
These experiments correspond to the main paper setting:
IR-only input 
random split 
scaffold-disjoint split 
complexity-controlled evaluation (mask_sum >= 3) 
pairwise MLP scorer 
sparse export + constrained decoding 
6.2 Strict training-vocabulary experiment
Use the same main pipeline, but build the vocabulary only from the training split and keep it fixed for export and decoding. This corresponds to the strict-vocabulary analysis in the paper.
6.3 Oracle-fragment experiment
Use the same decoding/evaluation setting while supplying ground-truth BRICS fragments during evaluation. This experiment is used to distinguish fragment-coverage error from connection-prediction error and corresponds to the oracle-fragment analysis in the paper.
6.4 Sparse export ablation
Run sparse export with different topk values, for example:
topk = 16 
topk = 32 
topk = 64 
and decode each setting separately. This matches the sparse export ablation reported in the paper.
6.5 Unfiltered vs filtered comparison
To reproduce the complexity-control comparison, run the same pipeline on:
the unfiltered scaffold-annotated dataset 
the filtered mask_sum >= 3 subset 
and compare reconstruction metrics. This corresponds to the complexity-controlled evaluation analysis in the paper.

7. Transformer comparison experiment
A separate Transformer-based comparison model is provided in:
ablation_transformer_stage2_model.py 
ablation_train_stage2_transformer.py 
ablation_export_stage2_transformer_npz.py 
Example training:
python ablation_train_stage2_transformer.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_global.tsv ^
  --ir_len 1024 ^
  --max_nodes 64 ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --device cuda ^
  --out best_transformer_random0.pt
Example export:
python ablation_export_stage2_transformer_npz.py ^
  --ckpt best_transformer_random0.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_transformer_random0_top32.npz ^
  --topk 32 ^
  --dtype float16
This branch is intended as a controlled architectural comparison under the same fragment-connection formulation.

8. Multimodal ablation experiment
Separate multimodal ablation scripts are provided in:
ablation_stage2_multimodal_dataset.py 
ablation_stage2_multimodal_model.py 
ablation_train_stage2_multimodal.py 
ablation_export_stage2_multimodal_npz.py 
The multimodal dataset parser supports auxiliary spectral inputs such as:
h_nmr_peaks 
c_nmr_peaks 
hsqc_nmr_peaks 
Example training:
python ablation_train_stage2_multimodal.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_global.tsv ^
  --ir_len 1024 ^
  --max_nodes 64 ^
  --use_h1 ^
  --use_c13 ^
  --use_hsqc ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --device cuda ^
  --out best_multimodal_random0.pt
Example export:
python ablation_export_stage2_multimodal_npz.py ^
  --ckpt best_multimodal_random0.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_multimodal_random0_top16.npz ^
  --topk 16 ^
  --dtype float16
These scripts support the modality-ablation analyses referenced in the main paper.

9. Supplementary peak-table experiment
For supplementary experiments based on peak-table style inputs, use:
si_peaktable_stage2_dataset.py 
si_peaktable_stage2_model.py 
si_peaktable_train.py 
si_peaktable_export_npz.py 
Example training:
python si_peaktable_train.py ^
  --train train_mask3_63k_random0_0.parquet ^
  --val val_mask3_63k_random0_0.parquet ^
  --vocab vocab_global.tsv ^
  --ir_len 1024 ^
  --max_nodes 64 ^
  --use_h1 ^
  --use_c13 ^
  --use_hsqc ^
  --epochs 60 ^
  --batch 64 ^
  --lr 1e-3 ^
  --device cuda ^
  --out best_si_peaktable_random0.pt
Example export:
python si_peaktable_export_npz.py ^
  --ckpt best_si_peaktable_random0.pt ^
  --data test_mask3_63k_random0_0.parquet ^
  --out npz_si_peaktable_random0_top32.npz ^
  --topk 32 ^
  --dtype float16
This branch is intended for supplementary multimodal peak-table experiments rather than the main IR-only paper setting.

10. Scaffold-level failure analysis for SI
Use:
python analyze_scaffold_failures.py ^
  --pred_csv pred_random0_top16.csv ^
  --test_parquet test_mask3_63k_random0_0.parquet ^
  --outdir scaffold_failure_random0
This script summarizes scaffold-level Top-1 failure counts, Top-1 failure rates, Top-5 failure rates, and typical failed cases. It is used for the scaffold-level failure analyses reported in the Supporting Information.

11. Mapping between paper sections and code
Main paper
preprocessing and filtering
make_scaffold_dataset.py, make_mask_sum_csv.py, filter_dataset_by_mask_sum_keep_scaffold.py 
split generation
random_split_from_scaffold_ds.py, scaffold_kfold_split.py 
main IR-only model
stage2_brics_dataset.py, stage2_brics_model.py, train_stage2_brics.py 
sparse export and decoding
export_stage2_logits_to_npz_dataset.py, denovo_decode_eval_from_npz_fast_v1.py 
main figure drawing
draw_ir_framework_figure.py 
Ablation / supplementary experiments
Transformer comparison
ablation_transformer_stage2_model.py, ablation_train_stage2_transformer.py, ablation_export_stage2_transformer_npz.py 
multimodal ablation
ablation_stage2_multimodal_dataset.py, ablation_stage2_multimodal_model.py, ablation_train_stage2_multimodal.py, ablation_export_stage2_multimodal_npz.py 
peak-table SI experiments
si_peaktable_stage2_dataset.py, si_peaktable_stage2_model.py, si_peaktable_train.py, si_peaktable_export_npz.py 
scaffold failure SI
analyze_scaffold_failures.py 

12. Notes
The main paper claims should be reproduced with the IR-only Stage-2 MLP pipeline, not with the Transformer or multimodal branches. 
Transformer and multimodal scripts are intended for controlled comparison and supplementary analysis. 
Supporting Information includes scaffold-level failure portraits and supplementary peak-table analyses. 

Citation
If you use this repository, please cite the paper and the processed dataset release associated with this project.

