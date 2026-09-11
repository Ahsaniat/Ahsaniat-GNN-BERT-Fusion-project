
# GNN-Based BERT for Understanding Context from Music



## Overview

This project implements a supervised multimodal neural network system for understanding musical context by combining textual semantics with graph-structured audio representations.

The system uses:

- **DistilBERT** to encode natural-language music descriptions.
- **GraphSAGE** to encode graph representations constructed from short-time audio segments.
- **Early Concatenation, Cross-Attention, and Gated Fusion** to combine text and audio representations.
- **InfoNCE contrastive learning** to align captions and audio graphs in a shared embedding space.
- **A mel-spectrogram CNN baseline** for comparison with the graph-based audio model.

The project evaluates four main tasks:

1. BERT-based multi-label music-context classification.
2. GraphSAGE-based audio classification.
3. GNN–BERT multimodal fusion.
4. Contrastive caption–audio retrieval.

Two datasets are used in the final experiments:

- **MusicCaps** for multimodal context tagging and audio-text retrieval.
- **GTZAN** for audio-only genre classification.

The final pipeline includes leakage-aware train/validation/test splitting, training-only normalization, multi-seed evaluation, per-label threshold calibration, validation-based model selection, persistent Google Drive storage, automatic checkpointing, and reproducible evaluation.

---

## Project Motivation

Music contains several levels of contextual information at the same time. A recording may communicate:

- genre,
- mood,
- instrumentation,
- tempo,
- recording quality,
- vocal characteristics,
- harmonic structure,
- production style,
- and other semantic characteristics.

Audio-only models can represent acoustic information but may not directly capture the rich semantic descriptions associated with a piece of music. Text models can understand semantic descriptions, but they do not directly represent temporal or structural relationships in the audio signal.

This project studies whether graph-based audio representations and contextual language representations can complement one another.

A music clip is represented conceptually as

$$ \[T = (X_{\text{audio}}, X_{\text{text}}, G, y)\] $$

where:

- $X_{\text{audio}}$ is the audio signal or derived acoustic representation,
- $X_{\text{text}}$ is a caption,
- $G=(V,E)$ is a graph constructed from audio segments,
- $y$ is the target context-label vector.

The text representation is produced by DistilBERT, while the audio graph is processed with GraphSAGE. Their learned representations can then be used independently or fused for prediction.

---

## Main Tasks

### Task 1 — DistilBERT Music-Context Classifier

The first task provides a strong text-only baseline.

A MusicCaps caption is tokenized and processed using:

```text
distilbert-base-uncased
````

The contextual representation of the first token is projected into the project embedding space and passed to a multi-label classification head.

For \(K\) music-context tags:

$$
\hat{y} = \sigma(Wt+b)
$$

where \(t\) is the DistilBERT representation.

Training uses class-weighted binary cross entropy.

This baseline is particularly important because MusicCaps captions already contain substantial semantic information about the corresponding aspect labels.

---

### Task 2 — GraphSAGE Audio Encoder

The second task represents each audio recording as a graph.

Each node corresponds to a short audio segment and contains acoustic features extracted from that segment.

Edges represent:

* temporal adjacency,
* acoustic similarity,
* and nearest-neighbor similarity relationships.

GraphSAGE performs message passing between connected audio segments.

For layer \(l\):

$$h_i^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{CONCAT}\left[h_i^{(l)}, \text{MEAN}_{j\in\mathcal{N}(i)}h_j^{(l)}\right]\right)$$
After the final GraphSAGE layer, graph-level mean pooling produces an audio representation.

The graph encoder is evaluated in two settings:

* MusicCaps multi-label context prediction.
* GTZAN genre classification.

---

### Task 3 — GNN–BERT Multimodal Fusion

Three fusion approaches are implemented.

#### Early Concatenation

The pooled GraphSAGE representation and DistilBERT representation are concatenated:

$$
z=[g;t]
$$

and passed through a classifier.

#### Cross-Attention

The graph representation is used as a query over contextual DistilBERT token representations.

$$
A =
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d}}
\right)
$$

The attended textual representation is combined with the graph representation for classification.

#### Gated Fusion

A learned gate determines the contribution of audio and text features.

Conceptually:

$$
\alpha = \sigma(W[g;t])
$$

$$
z = \alpha \odot t + (1-\alpha)\odot g
$$

after projection into a common representation space.

This allows the model to dynamically weight the two modalities instead of forcing equal contribution.

---

### Task 4 — Contrastive Audio–Text Retrieval

The fourth task trains GraphSAGE and DistilBERT representations in a shared embedding space using InfoNCE.

For graph representation \(g_i\) and corresponding caption representation \(t_i\):

$$\mathcal{L}_{NCE} = -\log \frac{\exp(\text{sim}(g_i,t_i)/\tau)}{\sum_j \exp(\text{sim}(g_i,t_j)/\tau)}$$
where cosine similarity is used.

The trained representations are evaluated for:

* Caption → Audio retrieval.
* Audio → Caption retrieval.

Metrics include:

* Recall@1,
* Recall@5,
* Recall@10.

---

# Datasets

## MusicCaps

MusicCaps contains **5,521 metadata entries** containing:

* YouTube video identifiers,
* expert-written natural-language captions,
* and aspect lists describing musical characteristics.

Direct downloading of the complete dataset is not reliable because many original YouTube videos are now:

* deleted,
* private,
* region restricted,
* age restricted,
* authentication restricted,
* or otherwise unavailable.

The data pipeline therefore maintains a persistent unavailable-video cache and a known blocked-video list so that failed IDs are not repeatedly queried.

The final experimental run obtained:

| Split      | Examples |
| ---------- | -------: |
| Train      |      396 |
| Validation |       84 |
| Test       |       88 |
| **Total**  |  **568** |

The supervised label vocabulary contains the **50 most frequent aspect tags** satisfying the minimum training-support requirement.

The vocabulary is created using the **training split only**.

Validation and test labels do not influence vocabulary construction.

---

## GTZAN

GTZAN contains 1,000 music tracks distributed across ten genres:

* blues,
* classical,
* country,
* disco,
* hiphop,
* jazz,
* metal,
* pop,
* reggae,
* rock.

One track failed preprocessing in the final pipeline, producing **999 processed tracks**.

The final split is:

| Split      | Tracks |
| ---------- | -----: |
| Train      |    699 |
| Validation |    150 |
| Test       |    150 |

Splitting is performed at the **track level**.

The same split definitions are used by the GraphSAGE and CNN models to maintain a fair comparison.

---

# Audio Preprocessing

All audio is resampled to:

```text
22,050 Hz
```

For MusicCaps, the available approximately 10-second excerpts are segmented using:

```text
Segment length : 2.0 seconds
Hop length     : 0.5 seconds
```

Each segment becomes one graph node.

## Node Features

For each audio segment the system extracts statistics from:

* 64-bin log-mel spectrogram,
* 12 chroma features,
* 20 MFCC coefficients,
* RMS energy,
* spectral centroid,
* spectral rolloff,
* zero-crossing rate.

Mean and standard deviation statistics are used where applicable.

The resulting node representation has:

```text
200 features
```

Therefore:

$$
x_i \in \mathbb{R}^{200}
$$

for node \(i\).

---

# Graph Construction

Each audio recording is converted into a segment graph.

## Temporal Edges

Adjacent segments are connected bidirectionally:

```text
i <-> i + 1
```

This preserves local temporal structure.

## Similarity Edges

Cosine similarity is calculated between node feature vectors.

Additional edges are created using:

```text
Cosine similarity threshold : 0.55
Maximum similar neighbors   : 4
```

This creates non-local links between acoustically related segments.

As a result, the graph contains both:

* local chronological structure,
* recurrence-like similarity connections.

For the 568 processed MusicCaps clips, the average graph contains approximately:

```text
17 nodes
92.7 directed edges
```

---

# Feature Normalization

Graph normalization statistics are calculated **only from training graphs**.

The resulting statistics are then applied unchanged to:

* validation graphs,
* test graphs.

This prevents information from validation or held-out examples from leaking into training.

---

# Model Architecture

## DistilBERT Encoder

Base model:

```text
distilbert-base-uncased
```

Maximum token length:

```text
128
```

Hidden size:

```text
768
```

The entire DistilBERT backbone is fine-tuned during supervised and contrastive training.

Its output is projected to the common multimodal representation dimension.

---

## GraphSAGE Encoder

The audio encoder contains four GraphSAGE message-passing layers.

High-level flow:

```text
Audio
  |
  v
Segment Feature Extraction
  |
  v
Graph Construction
  |
  v
GraphSAGE Layer 1
  |
  v
GraphSAGE Layer 2
  |
  v
GraphSAGE Layer 3
  |
  v
GraphSAGE Layer 4
  |
  v
Global Mean Pooling
  |
  v
Graph Embedding
```

The GraphSAGE network is trained end-to-end rather than being used as a frozen or randomly initialized feature generator.

---

## GNN–BERT Pipeline

```text
                         Music Clip
                             |
             +---------------+---------------+
             |                               |
             v                               v
          Caption                           Audio
             |                               |
             v                               v
        DistilBERT                    Segment Features
             |                               |
             |                               v
             |                         Structure Graph
             |                               |
             |                               v
             |                           GraphSAGE
             |                               |
             +---------------+---------------+
                             |
                             v
                    Multimodal Fusion
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
            Early       Cross-Attention    Gated
         Concatenation      Fusion          Fusion
              |              |              |
              +--------------+--------------+
                             |
                             v
                  Multi-Label Prediction
```

---

# Training Configuration

The final experiments were executed on a Tesla T4 GPU.

```text
GPU                         : Tesla T4
VRAM                        : approximately 14.56 GB
Supervised batch size       : 64
Evaluation batch size       : 128
Contrastive batch size      : 96
Maximum text length         : 128
Mixed precision             : enabled
Gradient clipping           : 1.0
Warmup                      : 8%
Learning-rate schedule      : cosine decay
```

## Learning Rates

```text
DistilBERT parameters       : 2e-5
Graph/fusion/head params    : 8e-4
```

Optimizer:

```text
AdamW
```

Weight decay:

```text
1e-4
```

---

# Class Imbalance Handling

MusicCaps is a strongly imbalanced multi-label dataset.

Positive-class weights are therefore calculated from the training split independently for each tag.

The maximum positive weight is capped at:

```text
30
```

The classification objective uses weighted binary cross entropy.

---

# Validation Threshold Calibration

A single probability threshold is not necessarily optimal for all music tags.

A separate threshold \(\tau_k\) is therefore selected for every tag using the validation set.

Candidate values are searched from:

```text
0.10 to 0.90
```

with step:

```text
0.025
```

The threshold maximizing validation F1 for each tag is selected.

These thresholds are frozen before held-out test evaluation.

The test set is never used for threshold selection.

---

# Early Stopping and Model Selection

Early stopping is based on validation Macro-F1.

```text
Patience: 7 epochs
```

Validation data is used for:

* checkpoint selection,
* threshold calibration,
* architecture selection,
* ensemble selection.

Held-out test metrics are calculated only after the corresponding validation decisions have been finalized.

---

# Multi-Seed Evaluation

The main models are evaluated using three random seeds:

```text
17
42
1337
```

Results are therefore reported as mean ± standard deviation where applicable.

This reduces dependence on a single random initialization.

---

# Evaluation Metrics

## Multi-Label Tagging

The MusicCaps classification task reports:

* Macro-F1,
* Micro-F1,
* Macro-AUPRC.

### Macro-F1

$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$
This assigns equal importance to each tag.

### Micro-F1

Micro-F1 pools all tag decisions before computing the F1 score.

### Macro-AUPRC

The precision-recall area is calculated independently for each tag and averaged.

This is particularly useful for imbalanced multi-label classification.

---

## Genre Classification

GTZAN reports:

* Accuracy,
* Macro-F1.

---

## Retrieval

Contrastive retrieval reports:

* R@1,
* R@5,
* R@10,

for both:

```text
Caption -> Audio
Audio -> Caption
```

---

# Experimental Results

## MusicCaps Multi-Label Classification

Three-seed held-out results:

| Model               |          Macro-F1 |          Micro-F1 |       Macro-AUPRC |
| ------------------- | ----------------: | ----------------: | ----------------: |
| BERT-only           | **0.475 ± 0.013** | **0.518 ± 0.011** | **0.581 ± 0.002** |
| GNN-only            |     0.138 ± 0.002 |                 — |                 — |
| Early Concatenation |     0.405 ± 0.006 |                 — |                 — |
| Cross-Attention     |     0.364 ± 0.018 |                 — |                 — |
| Gated Fusion        | **0.411 ± 0.033** |                 — |                 — |

The strongest multimodal architecture was **Gated Fusion**, while the strongest overall model remained **BERT-only**.

---

## Validation-Selected Final MusicCaps Model

The best model according to validation Macro-F1 was:

```text
Architecture : BERT-only
Seed         : 42
Validation Macro-F1 : 0.6102
```

Its held-out test performance was:

| Metric      |      Score |
| ----------- | ---------: |
| Macro-F1    | **0.4917** |
| Micro-F1    | **0.5129** |
| Macro-AUPRC | **0.5841** |

A validation-weighted multimodal ensemble obtained lower validation Macro-F1 than the selected BERT checkpoint and was therefore rejected before held-out model selection.

---

# Why the Text Baseline Performs Best on MusicCaps

The MusicCaps supervised task predicts aspect tags using a natural-language caption describing the same music clip.

Many target concepts are directly represented in caption language, including terms related to:

* recording quality,
* instrumentation,
* tempo,
* mood,
* vocals,
* genre,
* production characteristics.

Examples include:

```text
low quality
passionate
male vocal
slow tempo
acoustic guitar
noisy
instrumental
```

DistilBERT therefore receives a high-bandwidth semantic description that is strongly aligned with the target tag vocabulary.

The audio branch must infer the same concepts from approximately ten seconds of audio using hand-crafted segment-level acoustic features.

The result does not imply that graph modeling is ineffective. Instead, it demonstrates that the MusicCaps caption-to-aspect proxy task is heavily semantic and strongly favors the text modality.

GTZAN provides the more direct test of the graph encoder as an audio representation.

---

# Contrastive Retrieval Results

The final contrastive model was evaluated over the 88-example held-out MusicCaps test split.

| Direction       |        R@1 |        R@5 |       R@10 |
| --------------- | ---------: | ---------: | ---------: |
| Caption → Audio |  **4.55%** | **22.73%** | **44.32%** |
| Audio → Caption | **10.23%** | **28.41%** | **44.32%** |

For 88 candidates, random R@5 is approximately:

$$
\frac{5}{88}\approx5.68\%
$$

Both retrieval directions therefore show non-trivial learned cross-modal alignment.

---

# GTZAN Genre Classification

The GraphSAGE encoder is compared against a CNN operating on mel-spectrograms.

Three-seed results:

| Model               |          Accuracy |            Macro-F1 |
| ------------------- | ----------------: | ------------------: |
| GraphSAGE           | **70.22 ± 1.91%** | **0.7020 ± 0.0159** |
| Mel-Spectrogram CNN | **71.78 ± 2.57%** | **0.7029 ± 0.0269** |

The CNN has a small average accuracy advantage, while the models obtain essentially the same mean Macro-F1.

GraphSAGE also shows lower Macro-F1 variance across the three seeds.

These results demonstrate that the segment-graph representation is competitive with a conventional spectrogram-based neural network for audio-only genre classification.


# Installation

The project is designed primarily for Google Colab with GPU acceleration.

A Python environment can also be created locally.

## Python Version

Recommended:

```text
Python 3.10+
```

## Core Dependencies

The project uses:

```text
torch
torchvision
torchaudio
torch-geometric
transformers
datasets
librosa
numpy
pandas
scikit-learn
scipy
matplotlib
networkx
yt-dlp
iterative-stratification
tqdm
```

Install dependencies using:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers datasets
pip install librosa numpy pandas scipy scikit-learn matplotlib networkx tqdm
pip install yt-dlp iterative-stratification
```

FFmpeg is required for audio extraction.

On Ubuntu/Colab:

```bash
apt-get update
apt-get install -y ffmpeg
```

---

# Google Colab Setup

The main experimental notebook is:

```text
notebooks/demo_context.ipynb
```

The notebook mounts Google Drive at the beginning of execution.

The persistent project directory is:

```text
/content/drive/MyDrive/CSE425_GNN_BERT_Music_Context/
```

All expensive or important artifacts are stored beneath this location.

This includes:

```text
audio downloads
processed graphs
dataset splits
failed-video cache
HuggingFace model cache
PyTorch cache
training checkpoints
resume checkpoints
evaluation metrics
plots
runtime logs
```

This design allows interrupted Colab sessions to reuse completed preprocessing and training outputs instead of rebuilding them from scratch.

---

# Persistent MusicCaps Downloader

Because MusicCaps references external YouTube videos whose availability changes over time, the downloader includes failure-aware caching.

The pipeline:

1. Loads MusicCaps metadata.
2. Normalizes YouTube IDs.
3. Removes known blocked IDs.
4. Checks previously downloaded WAV files.
5. Checks the persistent failure cache.
6. Downloads only unresolved candidate videos.
7. Extracts the required audio interval.
8. Stores successful audio in Google Drive.
9. Records permanent failures.
10. Stops once the configured usable-pair target is reached or candidates are exhausted.

Known unavailable IDs are filtered before `yt-dlp` execution.

New failures are stored in:

```text
data/splits/musiccaps_failed_cache.json
```

This avoids repeatedly spending runtime on the same deleted, private, restricted, or unavailable videos.

The final experiment used the **568 valid paired clips actually available in the completed run**, rather than assuming that all MusicCaps metadata entries were downloadable.

---

# Training Checkpoints

Validation-best weights are saved throughout training.

Main checkpoint files include:

```text
musiccaps_bert_only_best.pt
musiccaps_gnn_only_best.pt
musiccaps_early_best.pt
musiccaps_cross_attention_best.pt
musiccaps_gated_best.pt
musiccaps_contrastive_best.pt
gtzan_gnn_best.pt
gtzan_cnn_best.pt
```

Resume checkpoints are also stored persistently so long-running experiments can continue after a runtime interruption.

---

# Output Artifacts

The complete experiment produces:

## Metrics

```text
results/FINAL_DASHBOARD.json
results/musiccaps_multiseed_metrics.json
results/musiccaps_retrieval.json
results/musiccaps_fusion_ensemble.json
results/gtzan_metrics.json
```

## Visualizations

The project generates:

* tag-support plots,
* learning curves,
* model-ablation comparisons,
* precision-recall curves,
* graph visualizations,
* node-feature matrices,
* t-SNE multimodal embeddings,
* cross-attention token visualizations,
* contrastive-training curves,
* retrieval similarity heatmaps,
* GTZAN genre distributions,
* GTZAN confusion matrices,
* mel-spectrogram examples,
* qualitative probability profiles.

---

# Example Qualitative Prediction

A held-out MusicCaps example may contain true labels such as:

```text
low quality
passionate
emotional
noisy
```

while the model may predict:

```text
low quality
passionate
emotional
noisy
shimmering hi hats
```

The complete test export retains prediction probabilities rather than only thresholded labels, enabling detailed error analysis.

---

# Runtime Logging

Training and preprocessing activity is written to:

```text
results/runtime.log
```

Logs include information such as:

* preprocessing progress,
* successful/failed audio downloads,
* graph construction,
* current epoch,
* training loss,
* validation Macro-F1,
* validation Micro-F1,
* AUPRC,
* learning rate,
* checkpoint updates,
* elapsed time,
* CUDA memory allocation,
* CUDA peak memory,
* model-selection decisions,
* final artifact locations.

---

# Reproducibility

The project controls important sources of experimental variation.

Main reproducibility mechanisms include:

* fixed random seeds,
* persistent split definitions,
* training-only feature normalization,
* training-only tag vocabulary construction,
* multi-label stratification,
* validation-only threshold calibration,
* validation-only checkpoint selection,
* validation-only architecture selection,
* validation-only ensemble selection,
* three-seed experiments,
* persistent checkpoints,
* complete held-out prediction export,
* saved metrics and figures.

The held-out MusicCaps and GTZAN test sets are not used to optimize thresholds or select checkpoints.

---

# Model Comparison Summary

The final experiments demonstrate different strengths for different modalities.

### MusicCaps

Text is the strongest modality for the supervised aspect-tag task.

```text
BERT-only > multimodal fusion > GNN-only
```

This is consistent with the strong semantic overlap between captions and aspect labels.

### GTZAN

GraphSAGE is competitive with a mel-spectrogram CNN.

```text
GraphSAGE Macro-F1 : 0.7020 ± 0.0159
Mel-CNN Macro-F1   : 0.7029 ± 0.0269
```

This shows that graph-structured segment representations can provide useful audio-only representations.

### Retrieval

Contrastive training creates measurable alignment between audio graphs and language representations.

```text
Caption -> Audio R@5 : 22.73%
Audio -> Caption R@5 : 28.41%
```

---

# Limitations

The final system has several limitations.

## MusicCaps Availability

Only 568 valid paired clips were available in the completed run.

The usable subset depends on external YouTube availability and therefore may not be representative of all 5,521 metadata entries.

## Dataset Size

The final MusicCaps held-out test set contains only 88 examples.

With 50 imbalanced labels, per-tag estimates and calibrated thresholds can have substantial statistical variance.

## Acoustic Representation

The graph nodes use hand-crafted acoustic features rather than a large pretrained audio representation.

A pretrained audio encoder may provide substantially richer node features.

## Graph Readout

Global mean pooling compresses the entire graph into one vector before most fusion operations.

This may remove fine temporal details useful for instrumentation and production tags.

## Proxy-Task Semantics

MusicCaps captions and aspect lists describe the same music clips.

The supervised tagging experiment therefore contains substantial semantic alignment between text input and labels.

It should not be interpreted as evidence that text is universally superior to audio for music understanding.

## GTZAN

GTZAN is a relatively small benchmark and has known dataset limitations.

The reported experiment is a controlled comparison within the implemented split rather than a claim of state-of-the-art genre classification.

## Emotion Regression

DEAM valence/arousal regression was not included in the final completed experiment because no correctly aligned DEAM audio/annotation pipeline was executed.

No synthetic or acoustically inferred values are reported as emotion ground truth.

---

# Future Development

Several extensions could strengthen the multimodal architecture.

### Pretrained Audio Representations

Hand-crafted segment features could be replaced or augmented with embeddings from a pretrained audio model.

The resulting embeddings could initialize graph nodes before GraphSAGE message passing.

### Hierarchical Music Graphs

Instead of representing only short segments, graphs could contain several levels:

```text
frames
  ->
segments
  ->
musical sections
  ->
whole-track representation
```

This could capture both local and long-range structure.

### Token-Level Audio–Text Fusion

Instead of globally pooling the graph before fusion, audio node embeddings could interact directly with individual text tokens.

This would allow cross-attention to learn fine-grained segment-to-language correspondences.

### Contrastive Pretraining

The graph and text encoders could first be contrastively pretrained and then fine-tuned for supervised tagging.

This may produce stronger multimodal alignment when labeled paired data is limited.

### Larger Paired Dataset

A larger locally stored or officially distributed paired audio-caption collection would reduce dependence on changing YouTube availability.

### Improved Retrieval Objectives

Retrieval could incorporate multiple-positive contrastive objectives for captions or clips that are semantically similar.

### More Stable Threshold Calibration

Larger validation sets, nested validation, or cross-validation could provide more stable per-tag threshold estimation.

---

# Key Findings

The completed experiments support three main findings.

1. **Text semantics dominate the MusicCaps aspect-tag proxy task.**

   Fully fine-tuned DistilBERT produced the strongest supervised tagging performance.

2. **Graph-based audio modeling is competitive for audio-only genre recognition.**

   GraphSAGE achieved almost the same mean Macro-F1 as the mel-spectrogram CNN on GTZAN.

3. **Audio graphs and captions can be aligned contrastively.**

   Caption-to-audio and audio-to-caption retrieval both substantially exceeded random R@5 performance.

The experiments therefore show that graph-structured representations can encode useful musical information while also emphasizing the importance of comparing multimodal systems against strong unimodal baselines.

---

# Final Results Summary

```text
MusicCaps usable paired examples     : 568
MusicCaps train / val / test         : 396 / 84 / 88
MusicCaps supervised tags            : 50

BERT-only Macro-F1                   : 0.475 ± 0.013
BERT-only Micro-F1                   : 0.518 ± 0.011
BERT-only Macro-AUPRC                : 0.581 ± 0.002

Best validation-selected test:
Macro-F1                             : 0.4917
Micro-F1                             : 0.5129
Macro-AUPRC                          : 0.5841

Best multimodal family:
Gated Fusion Macro-F1                : 0.411 ± 0.033

Caption -> Audio R@1                 : 4.55%
Caption -> Audio R@5                 : 22.73%
Caption -> Audio R@10                : 44.32%

Audio -> Caption R@1                 : 10.23%
Audio -> Caption R@5                 : 28.41%
Audio -> Caption R@10                : 44.32%

GTZAN GraphSAGE Accuracy             : 70.22 ± 1.91%
GTZAN GraphSAGE Macro-F1             : 0.7020 ± 0.0159

GTZAN Mel-CNN Accuracy               : 71.78 ± 2.57%
GTZAN Mel-CNN Macro-F1               : 0.7029 ± 0.0269
```

---

## Technologies

```text
Python
PyTorch
PyTorch Geometric
HuggingFace Transformers
DistilBERT
GraphSAGE
Librosa
Scikit-learn
NumPy
Pandas
Matplotlib
NetworkX
yt-dlp
FFmpeg
Google Colab
Google Drive
CUDA
```

---

## License and Dataset Notice

This repository contains source code, model definitions, experiment outputs, and derived artifacts.

MusicCaps and GTZAN remain subject to their respective dataset terms and original-source conditions.

MusicCaps metadata references externally hosted YouTube content. Raw MusicCaps audio should therefore not be treated as a dataset redistributed by this project.

The project stores only audio obtained during preprocessing in the user's own persistent runtime environment and records source identifiers for reproducibility.



