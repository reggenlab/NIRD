## NIRD

**NIRD (Network Inference by Reduced Dimension)** is a Linux command-line tool for large-scale Gene Regulatory Network (GRN) inference from single-cell and bulk RNA-seq data. NIRD applies matrix factorization–based dimensionality reduction techniques to gene expression profiles in order to capture both linear and non-linear regulatory relationships that are difficult to infer from sparse, high-dimensional data.

By operating in a reduced feature space, NIRD enables efficient GRN inference while preserving biologically meaningful regulatory signals. The inferred networks are evaluated using Area Under the Curve (AUC)–based performance metrics, and NIRD outputs complete gene-by-gene interaction matrices representing regulatory relationships across cellular populations.

Please see our manuscript for a detailed description of the methodology and benchmarking results.

---

## Dependencies

NIRD is tested to work under **Python 3.8**.  
The required dependencies are listed below:

- python 3.8  
- numpy 1.23.1  
- pandas 1.5.3  
- scikit-learn 1.1.1  
- scipy 1.9.3  
- matplotlib 3.6.3  
- seaborn 0.12.2  
- distributed 2022.7.0  
- nimfa 1.4.0  
- networkx 2.8.8  
- arboreto 0.1.6  

---

## Installation

Installing NIRD within a Conda environment is recommended.

### Step 1: Create a Conda environment

```bash
conda create -n nird python=3.8 pip -y
```
OR

```bash
conda env create -f nird.yml
```

### Step 2: Activate conda environment

```bash
conda activate nird
```

### Step 3: Install required dependencies

Install all required libraries using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

---

## Input Data Format

NIRD expects a gene expression matrix in CSV or TSV format, where:

- Rows represent samples (e.g. GSM IDs)
- Columns represent genes
- Values represent gene expression levels (TPM, FPKM, or raw counts)

---

## Usage

Navigate to the NIRD working directory before running the tool.

### 1. Single Expression Mode

Use this mode when only one expression dataset is available.

```bash
python NIRD.py --datasets single_expr --file1 MF_Datasets/mESC/smartSeq.csv --outdir inferred_networks
```

### 2. Double Expression Mode

Use this mode to infer and compare GRNs from two expression datasets.

```bash
python NIRD.py --datasets double_expr --file1 MF_Datasets/mESC/dropSeq.csv --file2 MF_Datasets/mESC/smartSeq.csv --outdir inferred_networks
```

### 3. Gold Data Mode

Use this mode when expression data, transcription factor data, and a gold standard network are available.

```bash
python NIRD.py --datasets gold_data --expr_file MF_Datasets/dream5/net2/dream5_net2_expression_data.tsv --tf_file MF_Datasets/dream5/net2/dream5_net2_transcription_factors.tsv --gold_file MF_Datasets/dream5/net2/dream5_net2_gold.tsv --outdir inferred_networks
```

### 4. NIRD_Velo Mode

Use this mode when time-course expression or RNA velocity data are available.

```bash
python NIRD_Velo.py --file1 MF_Datasets/transcription_velocity/00h_time_course_expr.csv --file2 MF_Datasets/transcription_velocity/0th_hr_endo_RNA_Velo.csv --outdir inferred_networks
```

---

## Outputs

Depending on the selected mode, NIRD generates:

- Gene-by-gene inferred regulatory network matrices
- Ranked edge lists representing regulatory interactions
- Evaluation plots and AUC scores (if evaluation is enabled)

All outputs are saved in the user-specified output directory.

---

## Datasets

Demo datasets used for benchmarking and evaluation in NIRD include:

- Single-cell RNA-seq datasets
- Bulk RNA-seq datasets
- Gold standard regulatory networks (when applicable)

Dataset details are provided in the manuscript.

---

## Citing NIRD

If you find NIRD useful in your research, please consider citing our work:
[Citation information to be added after publication]

---


