# Gravity Falls Jigsaw Reconstruction (Milestone 1 + Milestone 2)

Computer-vision pipeline for reconstructing square jigsaw puzzles from the Gravity Falls dataset.  
Milestone 1 generates puzzle-piece metadata (contours/corners/edges) and multiple tile “versions”.  
Milestone 2 evaluates versions, solves 2×2 robustly, and implements a 4×4 solver using seam costs + global search.

---

## Table of contents
- [1) Problem statement](#1-problem-statement)
- [2) Key idea](#2-key-idea)
- [3) Dataset & outputs](#3-dataset--outputs)
- [4) Milestone 1 pipeline](#4-milestone-1-pipeline)
  - [4.1 Preprocessing](#41-preprocessing)
  - [4.2 Contour extraction](#42-contour-extraction)
  - [4.3 Corner estimation](#43-corner-estimation)
  - [4.4 Edge representation](#44-edge-representation)
  - [4.5 Tile “versions”](#45-tile-versions)
- [5) Milestone 2 pipeline](#5-milestone-2-pipeline)
  - [5.1 Version evaluation](#51-version-evaluation)
  - [5.2 2×2 solver (ZNCC + shift + exhaustive)](#52-22-solver-zncc--shift--exhaustive)
  - [5.3 4×4 solver (ZNCC + shift + SA)](#53-44-solver-zncc--shift--sa)
- [6) Why 4×4 is harder than 2×2](#6-why-44-is-harder-than-22)
- [7) Current results & limitations](#7-current-results--limitations)
- [8) How each step contributes to the goal](#8-how-each-step-contributes-to-the-goal)
- [9) References](#9-references)

---

## 1) Problem statement

Given a set of shuffled square tiles for each image (2×2 and 4×4), reconstruct the original image arrangement.
Tiles are provided in multiple “versions” (e.g., color, grayscale, sharpened). The objective is to output the correct
permutation (tile placement in a grid).

---

## 2) Key idea

The project uses **edge/seam compatibility**:

- In **Milestone 1**, we build a clean geometric representation (contours, corners, edges) and produce improved tile
  versions for stronger matching signals.
- In **Milestone 2**, we compute a **pairwise seam cost** between tiles using **ZNCC** (Zero-mean Normalized Cross
  Correlation) and allow small **shifts** to handle slight misalignment. We then perform **global optimization** to
  select a consistent full-grid placement.

---

## 3) Dataset & outputs

### Expected output directory (Google Drive / Colab)

/content/drive/MyDrive/Gravity Falls/output/
puzzle_2x2/<image_id>/<version>/0.png 1.png 2.png 3.png
puzzle_4x4/<image_id>/<version>/0.png ... 15.png


### Tile versions
Milestone 1 produces multiple processed versions (the repo expects at least):
- `color` (BGR tiles for visualization)
- `sharpened` (grayscale sharpened tiles used for matching in Milestone 2)

---

## 4) Milestone 1 pipeline

Milestone 1 is responsible for converting raw puzzle imagery into:
1) tile crops (organized by image id/version), and  
2) metadata describing each piece (contours/corners/edges).

### 4.1 Preprocessing
**Goal:** convert the image into a form where piece boundaries are easy to detect.

Typical steps:
- convert to grayscale
- denoise (light blur if needed)
- thresholding (often Otsu)
- morphological cleanup (open/close) to remove holes and small artifacts

**Justification:** segmentation/contour extraction is highly sensitive to noise. Preprocessing increases the chance that
each puzzle piece becomes one clean connected component.

### 4.2 Contour extraction
**Goal:** detect each piece as a region boundary.

Approach:
- `cv2.findContours(..., RETR_EXTERNAL, ...)`
- filter contours by area

**Justification:** the contour gives a direct boundary representation and is the foundation for corners/edges.

### 4.3 Corner estimation
**Goal:** approximate each piece as a 4-corner polygon (since pieces are square tiles).

Approach:
- polygon approximation using `cv2.approxPolyDP`
- fallback strategies when the approximation is not exactly 4 points

**Justification:** corners provide a consistent way to define edges and their ordering (top/right/bottom/left).

### 4.4 Edge representation
**Goal:** represent each edge in a comparable, structured way.

Approach:
- sample points along the contour segment between corner pairs
- optionally resample to fixed-length point sequences

**Justification:** enables shape-based matching and provides a consistent geometric descriptor for edges, independent of
pixel intensities.

### 4.5 Tile “versions”
Milestone 1 generates several versions (e.g., grayscale, sharpened).  
In Milestone 2 we empirically found **`sharpened`** to give the strongest seam matching signal.

**Justification:** sharpening boosts boundary/line contrast, which improves correlation-based seam comparison.

---

## 5) Milestone 2 pipeline

### 5.1 Version evaluation
We first benchmark all versions using a **2×2** solver (small enough to evaluate exactly).  
For each version:
1) compute seam costs with a simple metric (initially MSE on edge bands),
2) solve 2×2 by exhaustive permutation search,
3) compute accuracy using `GROUND_TRUTH`.

Outcome: **`sharpened` performed best** and was selected for the main solver.

### 5.2 2×2 solver (ZNCC + shift + exhaustive)
This stage achieved **~29/30 correct** using:
- **edge bands** (BAND_WIDTH = 5)
- **ZNCC** similarity (robust to brightness/contrast shifts)
- **shift search** (max_shift ≈ 6) to tolerate small alignment errors
- exhaustive global search over 24 permutations

**Why ZNCC:** ZNCC is a standard normalized similarity for template matching and correlation-based comparison.  
For a template-matching overview and NCC explanation, see e.g. Roboflow’s template matching primer, and the classic fast
NCC reference. :contentReference[oaicite:0]{index=0}

### 5.3 4×4 solver (ZNCC + shift + SA)
For 4×4 we compute pairwise seam costs:
- `C_R[i, j]`: cost if tile `j` is to the **right** of tile `i`
- `C_B[i, j]`: cost if tile `j` is **below** tile `i`

Total grid energy sums all internal seams:
- horizontal seams: 4 rows × 3 seams
- vertical seams: 3 seams × 4 cols

Because 16! placements are infeasible to brute-force, we apply **Simulated Annealing**:
- initialize with a greedy placement
- repeatedly propose random swaps
- accept improvements; occasionally accept worse moves (temperature schedule) to escape local minima

**Justification:** SA is a standard method for large combinatorial optimization with many local minima. :contentReference[oaicite:1]{index=1}

---

## 6) Why 4×4 is harder than 2×2

2×2 is small enough to compute the true global optimum (24 permutations).

4×4 introduces:
- factorial explosion (16!)
- many locally good seams that create “correct blocks”
- ambiguous textures (cartoon regions can be repetitive)
- multiple near-optimal global layouts with similar total seam energy

This is why you often observe:
- correct 2-piece or 4-piece blocks,
- but those blocks are swapped or placed in the wrong global location.

Academic solvers for larger puzzles usually combine strong pairwise compatibilities with **global assembly strategies**
(component merging, tree/MST growth, or LP-based formulations) rather than relying only on swap-based SA.  
Examples:
- Gallagher (CVPR 2012) discusses greedy/component-based assembly. :contentReference[oaicite:2]{index=2}  
- Yu, Russell, Agapito propose a global LP formulation that is more robust to local minima. :contentReference[oaicite:3]{index=3}  
- Paikin & Tal (CVPR 2015) show scalable strategies for large square puzzles. :contentReference[oaicite:4]{index=4}  

---

## 7) Current results & limitations

### Achieved
- Version evaluation identified **`sharpened`** as the most reliable for seam matching.
- 2×2 solver is highly accurate (~29/30).

### Limitations (4×4)
- SA frequently converges to near-correct solutions with correct local blocks.
- Missing explicit modeling of:
  - border/corner constraints,
  - mutual-best neighbor consistency,
  - component-first assembly,
  - global position constraints (e.g., LP).

---

## 8) How each step contributes to the goal

- **Preprocessing (Milestone 1):** creates clean masks/regions so contours correspond to real tiles, not noise.
- **Contour extraction:** identifies each tile boundary; enables robust cropping and geometric metadata.
- **Edge representation (point sequences):** makes edges comparable and supports higher-level matching ideas (shape-based).
- **Tile versioning (sharpened):** strengthens edge contrast so pixel-based seam matching becomes more discriminative.
- **Edge band extraction (Milestone 2):** focuses comparison on boundaries where adjacency information exists.
- **ZNCC + shift:** robust seam similarity measure tolerant to brightness changes and minor misalignment.
- **Global search:**
  - 2×2: exhaustive ensures a true optimum.
  - 4×4: SA approximates the optimum but may get trapped without stronger global constraints.

---

## 9) References

1. S. Kirkpatrick, C. D. Gelatt Jr., M. P. Vecchi, “Optimization by Simulated Annealing,” *Science*, 1983. :contentReference[oaicite:5]{index=5}  
2. A. C. Gallagher, “Jigsaw Puzzles with Pieces of Unknown Orientation,” *CVPR*, 2012. :contentReference[oaicite:6]{index=6}  
3. R. Yu, C. Russell, L. Agapito, “Solving Jigsaw Puzzles with Linear Programming,” *BMVC 2016* / arXiv 2015. :contentReference[oaicite:7]{index=7}  
4. K. Briechle, U. D. Hanebeck, “Template Matching using Fast Normalized Cross Correlation,” *SPIE*, 2001. :contentReference[oaicite:8]{index=8}  
5. G. Paikin, A. Tal, “Solving Multiple Square Jigsaw Puzzles With Missing Pieces,” *CVPR*, 2015. :contentReference[oaicite:9]{index=9}  
6. Template matching / NCC primer (high-level overview). :contentReference[oaicite:10]{index=10}  

---

## Running notes (Colab)

1. Upload notebooks
2. Mount Google Drive (Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
3. Ensure `OUTPUT_ROOT` points to your Drive path.
4. Run Milestone 1 for tile generation
5. Run Milestone 2 for evaluation and solving
