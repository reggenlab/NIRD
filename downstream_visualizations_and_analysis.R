
# setwd("path/to/the/workdir")
# list.files()

# -----------------------------------------------------------------------------------------------------------------------
# code for downstream visualization of inferred network from NIRD
# -----------------------------------------------------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(tibble)

mat_to_edges <- function(intr_mat, min_weight = 0) {
  
  as.data.frame(intr_mat) %>%
    rownames_to_column("source") %>%
    pivot_longer(
      cols = -source,
      names_to = "target",
      values_to = "weight"
    ) %>%
    filter(
      source != target,
      weight > min_weight
    )
}

htc_intr_mat_PCA <- read.csv("inferred_networks/HTC_inferred_network_PCA.csv", row.names = 1)
edges_PCA_vis <- mat_to_edges(htc_intr_mat_PCA, min_weight = 0)

write.csv(
  edges_PCA_vis,
  "PCA_network_edges.csv",
  row.names = FALSE
)

library(igraph)
library(tidygraph)
library(ggraph)

# Keep top 0.1% edges (adjust as needed)
edges_filt <- edges_PCA_vis %>%
  filter(weight >= quantile(weight, 0.999))

dim(edges_filt)

g <- graph_from_data_frame(edges_filt, directed = FALSE)

# Compute node metrics
V(g)$strength <- strength(g, weights = E(g)$weight)
V(g)$degree   <- degree(g)

hub_cutoff <- quantile(V(g)$strength, 0.95)
V(g)$hub <- V(g)$strength >= hub_cutoff

tg <- as_tbl_graph(g) %>%
  activate(nodes) %>%
  mutate(
    strength = strength,
    degree   = degree,
    hub      = hub
  )

set.seed(123)

p <- ggraph(tg, layout = "fr") +
  
  # Edges
  geom_edge_link(
    aes(width = weight),
    alpha = 0.4,
    colour = "grey60"
  ) +
  
  # Nodes
  geom_node_point(
    aes(
      size = strength,
      color = hub
    ),
    alpha = 0.9
  ) +
  
  # Label hub genes only
  geom_node_text(
    aes(label = ifelse(hub, name, "")),
    repel = TRUE,
    size = 4,
    fontface = "bold"
  ) +
  
  scale_edge_width(range = c(0.2, 1.5)) +
  scale_size_continuous(range = c(2, 8)) +
  scale_color_manual(
    values = c("FALSE" = "#2c7fb8", "TRUE" = "#d7301f")
  ) +
  
  theme_void() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  
  ggtitle("Gene Regulatory Network (PCA-based)")

print(p)

ggsave(
  "Gene_Network_PCA.pdf",
  plot = p,
  width = 16,
  height = 10,
  dpi = 600
)


# ------------------------------------------------------------------------------------------------------------------------
# code for motif mapping and regulon inference
# ------------------------------------------------------------------------------------------------------------------------

library(tidyverse)
library(readr)
library(stringr)
library(openxlsx)

motifs <- read_tsv(
  "reference_data/motifs-v9-nr.hgnc-m0.001-o0.0.tbl",
  col_types = cols(.default = "c")
)

head(motifs)

motifs_human <- motifs %>%
  filter(str_detect(orthologous_species, "H. sapiens"))

# write_csv(motifs_human, "reference_data/motifs_homo_sapiens.csv")

# Load the inferred networks
intr_0hr_PCA <- read.csv("inferred_networks/0hr_inferred_network_PCA.csv", row.names = 1)
intr_0hr_GRN <- read.csv("inferred_networks/0hr_inferred_network_GrnBoost2.csv", row.names = 1)
intr_12hr_PCA <- read.csv("inferred_networks/12hr_inferred_network_PCA.csv", row.names = 1)
intr_12hr_GRN <- read.csv("inferred_networks/12hr_inferred_network_GrnBoost2.csv", row.names = 1)

# Load the expression data
expr_0hr <- read.csv("MF_Datasets/transcription_velocity/Top_2000_Genes_00h.csv", row.names = 1)
expr_12hr <- read.csv("MF_Datasets/transcription_velocity/Top_2000_Genes_12h.csv", row.names = 1)

common_genes <- intersect(colnames(expr_0hr), colnames(expr_12hr))

expr_0hr  <- expr_0hr[, common_genes]
expr_12hr <- expr_12hr[, common_genes]

expr_cells <- rbind(expr_0hr, expr_12hr)
expr_combined <- t(expr_cells)

meta <- data.frame(
  cell = rownames(expr_cells),
  timepoint = c(
    rep("0hr", nrow(expr_0hr)),
    rep("12hr", nrow(expr_12hr))
  ),
  row.names = rownames(expr_cells)
)

library(AUCell)

cells_rankings <- AUCell_buildRankings(
  exprMat = expr_combined,
  plotStats = TRUE,
  verbose = TRUE
)

get_top_edges <- function(intr_mat, top_pct = 0.20) {
  
  df <- as.data.frame(intr_mat) %>%
    dplyr::select(-any_of(c("TF", "Gene1", "gene", "source"))) %>%
    rownames_to_column("TF")
  
  edges <- df %>%
    pivot_longer(-TF, names_to = "target", values_to = "weight") %>%
    filter(
      TF != target,
      !is.na(weight),
      weight > 0
    )
  
  cutoff <- quantile(edges$weight, probs = 1 - top_pct, na.rm = TRUE)
  
  edges %>% filter(weight >= cutoff)
}

edges_PCA <- bind_rows(
  get_top_edges(intr_0hr_PCA),
  get_top_edges(intr_12hr_PCA)
)

edges_GRN <- bind_rows(
  get_top_edges(intr_0hr_GRN),
  get_top_edges(intr_12hr_GRN)
)

human_TFs <- read.csv("reference_data/motifs_homo_sapiens.csv")

edges_PCA2 <- edges_PCA %>% mutate(TF_upper = toupper(TF))
edges_GRN2 <- edges_GRN %>% mutate(TF_upper = toupper(TF))

human_TFs2 <- human_TFs %>%
  mutate(gene_upper = toupper(gene_name))

map_motifs <- function(edges, motifs) {
  edges %>%
    left_join(motifs, by = c("TF_upper" = "gene_upper")) %>%
    filter(
      !is.na(orthologous_identity),
      orthologous_species == "H. sapiens"
    )
}

mapped_PCA <- map_motifs(edges_PCA2, human_TFs2)
mapped_GRN <- map_motifs(edges_GRN2, human_TFs2)

build_regulons <- function(mapped_edges, exprMat) {
  
  genes_in_expr <- rownames(exprMat)
  
  split(mapped_edges$target, mapped_edges$TF) %>%
    lapply(function(x) intersect(x, genes_in_expr)) %>%
    Filter(function(x) length(x) >= 5, .)
}

geneSets_PCA <- build_regulons(mapped_PCA, expr_combined)
geneSets_GRN <- build_regulons(mapped_GRN, expr_combined)

auc_PCA <- AUCell_calcAUC(geneSets_PCA, cells_rankings)
auc_GRN <- AUCell_calcAUC(geneSets_GRN, cells_rankings)

regMat_PCA <- t(getAUC(auc_PCA))
regMat_GRN <- t(getAUC(auc_GRN))

library(Rtsne)
library(ggplot2)
library(patchwork)

set.seed(123)

tsne_PCA <- Rtsne(scale(regMat_PCA), perplexity = 30)
tsne_GRN <- Rtsne(scale(regMat_GRN), perplexity = 30)

df_PCA <- data.frame(
  TSNE1 = tsne_PCA$Y[,1],
  TSNE2 = tsne_PCA$Y[,2],
  timepoint = meta$timepoint
)

df_GRN <- data.frame(
  TSNE1 = tsne_GRN$Y[,1],
  TSNE2 = tsne_GRN$Y[,2],
  timepoint = meta$timepoint
)

p1 <- ggplot(df_PCA, aes(TSNE1, TSNE2, color = timepoint)) +
  geom_point(size = 2) +
  theme_classic() +
  ggtitle("t-SNE plot of hESC single-cells (PCA): 2 time points")

p2 <- ggplot(df_GRN, aes(TSNE1, TSNE2, color = timepoint)) +
  geom_point(size = 2) +
  theme_classic() +
  ggtitle("t-SNE plot of hESC single-cells (GrnBoost2): 2 time points")

p3 <- p1 | p2
ggsave("0hr_12hr_cluster_plot.jpg", p3, dpi = 600, height = 6, width = 12)







