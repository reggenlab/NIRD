## NIRD 

NIRD permits researchers to perceive the contribution of the regulatory components by transforming datasets into networks, using dimensionality reduction and assessing the overlapping structure with AUC metrics. This methodology can be applied to describe a number of biological situations, such as disease vs. control comparisons and temporal dynamics.

[Click here](http://reggen.iiitd.edu.in:1207/NIRD/) to know more.


## Primery Setup

Step-1: Create a conda environment

```shell
conda create -n NIRD python=3.10 -y
```
You can change python=3.10 to another version if needed.


Step-2: Activate the Environment

```shell
conda activate NIRD
```
If you're using this in a script or .bashrc, you may need to source Conda first:

```shell
source ~/anaconda3/etc/profile.d/conda.sh
conda activate NIRD
```
Step-3: Install Required Libraries

```shell
pip install numpy pandas seaborn matplotlib scipy scikit-learn distributed arboreto nimfa
```
 
 version for which it worked : 
python > 3.0  ;        
numpy : 1.23.1  ;     
pandas : 1.5.3   ;     
sklearn : 1.1.1   ;     
matplotlib : 3.9.2   ;    
seaborn : 0.13.2   ;    
scipy : 1.8.1   ;    
distributed : 2022.7.0 ;    
 argparse : 1.1  ;     
nimfa : 1.4.0  ;     

## Input Data format

The NIRD tool requires a gene expression matrix as input. This matrix should be in CSV or TSV format. The matrix rows should represent samples and columns should represent genes. Each cell should contain the expression value of a gene in a given sample (e.g., TPM, FPKM, or raw counts, depending on your preprocessing pipeline).


Apart from standard expression-expression (expr-expr) network inference, it also allows calculation of dependencies using RNA-velocity and expression. 

# RNA-velocity matrix
The transcription velocity matrix should be a CSV file where:
Rows represent cells or samples (with IDs like SRR2978582, SRR2978568, etc.)
Columns represent genes (e.g., LINC02593, NOC2L, etc.)
Values are real numbers (positive or negative), typically indicating velocity (rate of change of expression)


##  Running the NIRD tool

Once you’ve completed the primary setup, you’re ready to run the NIRD tool for network inference and evaluation.

The NIRD tool supports four different modes depending on the type of data available:

Single Expression Mode – If you have only one expression dataset.
Double Expression Mode – If you have two expression datasets and want to check network overlap.
Gold Data Mode – If you have expression data, transcription factor data, and gold standard networks.
NIRD_Velo Mode – If you have transcription velocity and time-course expression data to infer expression-expression and expression-velocity based networks.


#Step 1: Activate the Conda Environment

Before running any commands, ensure the NIRD environment is activated:


```shell
conda activate NIRD
```


If you're using this in a script or inside .bashrc, remember to source Conda first:

```shell
source ~/anaconda3/etc/profile.d/conda.sh conda activate NIRD
```

#Step 2: Set the Python Path
Update the PYTHONPATH to include the NIRD directory:

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/network_inference
```

#Step 3: Change to the Working Directory
Navigate to the project directory:

```shell
cd /network_inference/NIRD_main
```

#Step 4: Run the NIRD Tool
Use one of the following commands depending on your data type:

# 1. Single Expression Mode
Use this when you have only one expression dataset:

```shell
python NIRD.py \
--datasets single_expr \
--file1 /path/to/file1.csv \
--outdir /path/to/output/directory
```


# 2. Double Expression Mode
Use this when you have two expression datasets and want to check network overlap:

```shell
python NIRD.py \
--datasets double_expr \
--file1 /path/to/file1.csv \
--file2 /path/to/file2.csv \
--outdir /path/to/output/directory
```

# 3. Gold Data Mode
Use this when you have expression data, TF data, and gold standard data:

```shell
python NIRD.py \
--datasets gold_data \
--expr_file /path/to/expression_file.tsv \
--tf_file /path/to/transcription_factors.tsv \
--gold_file /path/to/gold_data.tsv \
--outdir /path/to/output/directory
```

# 4. NIRD_Velo Mode
Use this when you have transcription velocity and time-course expression data to infer expression-expression and expression-velocity based networks:

```shell
python NIRD_Velo.py \
--file1 /path/to/file1.csv \
--file2 /path/to/file2.csv \
--outdir /path/to/output/directory
```

# Step 5: Help & Arguments
If you're unsure about the available command-line options or want to check how to properly format your input arguments, you can always view the detailed usage information using:

```shell
python NIRD.py --help
python NIRD_Velo.py --help
```








