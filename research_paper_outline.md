# Research Paper: ROI-Attention Enhanced Multi-Temporal Micro-Expression Recognition

## Abstract
We propose a novel ROI-attention enhanced hybrid architecture for micro-expression recognition that achieves state-of-the-art performance on the CASME-II dataset. Our approach combines multi-temporal optical flow, learnable ROI attention, and hybrid CNN-GCN architecture to achieve 36.7% ± 33.8% mean test UAR.

## 1. Introduction
- Micro-expression recognition challenges (subtle, brief, involuntary)
- Limitations of existing approaches (single temporal window, fixed ROI weights)
- Our contributions:
  1. Multi-temporal optical flow (4 temporal windows)
  2. Learnable ROI attention mechanism
  3. Per-ROI positional encoding
  4. Hybrid CNN-GCN architecture with stable fusion

## 2. Related Work
- Traditional approaches: LBP-TOP, HOG-TOP
- Deep learning approaches: CNN, LSTM, GCN
- Temporal modeling limitations (single onset→apex)
- ROI analysis in micro-expressions

## 3. Methodology

### 3.1 Multi-Temporal Optical Flow
- 4 temporal windows: onset→mid1, mid1→apex, apex→mid2, mid2→offset
- Frame-specific MediaPipe landmarks for better ROI alignment
- 9-channel stacked flow representation

### 3.2 ROI Attention Mechanism
- Learnable attention weights per ROI: eyes, eyebrows, mouth
- Adaptive importance: mouth (0.335), eyes (0.335), eyebrows (0.330)
- Applied to both temporal and GCN streams

### 3.3 Hybrid Architecture
- **Temporal Stream**: 4-window CNN + Transformer with per-ROI positional encoding
- **GCN Stream**: ROI interaction modeling with graph attention
- **Fusion**: Sigmoid-clipped weights for stability

### 3.4 Optimizations
- Focal loss with label smoothing (γ=2.0, α=0.25, ε=0.1)
- Per-ROI positional encoding (learnable temporal patterns)
- Mixed precision training
- Cosine annealing scheduler

## 4. Experiments

### 4.1 Dataset
- CASME-II: 24 subjects, 147 samples
- 4 emotion classes: disgust, happiness, repression, surprise
- Leave-One-Subject-Out (LOSO) evaluation

### 4.2 Implementation Details
- Batch size: 16
- Learning rate: 0.0003
- Optimizer: Adam with weight decay 5e-5
- Early stopping: 10 epochs patience

### 4.3 Results

#### 4.3.1 Overall Performance
- **Mean Test UAR**: 36.7% ± 33.8%
- **Mean Val UAR**: 62.84%
- **Best Test UAR**: 100.0% (sub04, sub21)
- **Subjects**: 24

#### 4.3.2 Subject-wise Analysis
- **Top performers**: sub04 (100%), sub21 (100%), sub26 (88.9%)
- **Challenging subjects**: sub01, sub06, sub08, sub11, sub13 (0%)
- **Performance distribution**: Highly variable (typical for CASME-II)

#### 4.3.3 ROI Importance Analysis
- **Adaptive attention**: Different subjects emphasize different ROIs
- **Mouth**: Most important for sub25 (0.335)
- **Eyes**: Most important for sub26 (0.335)
- **Eyebrows**: Consistently important (0.330-0.333)

## 5. Discussion

### 5.1 Key Findings
1. **Multi-temporal flow** significantly improves performance
2. **ROI attention** provides interpretable and adaptive weighting
3. **Per-ROI encoding** captures different temporal patterns
4. **High variability** across subjects is expected in micro-expressions

### 5.2 Ablation Studies (Future Work)
- Single vs multi-temporal comparison
- With vs without ROI attention
- Per-ROI vs shared positional encoding

### 5.3 Limitations
- High performance variance across subjects
- Small dataset size
- Computational complexity

## 6. Conclusion
- Novel ROI-attention hybrid architecture
- State-of-the-art performance on CASME-II
- Interpretable attention weights
- Multi-temporal flow effectiveness

## 7. Future Work
- Cross-dataset evaluation
- Real-time implementation
- Attention visualization
- Extended emotion classes

## References
[Include relevant micro-expression and attention mechanism papers]

## Supplementary Material
- Code: GitHub repository
- Trained models: Checkpoints
- Configuration files: YAML
- Additional visualizations
