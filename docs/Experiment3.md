# Experiment 3: Low-Rank Approximation Analysis

This experiment analyzes the behavior of low-rank approximations of trained bilinear models, validating **Theorem 4.1** from "On the Theoretical Advantages of Bilinear Similarities in Dense Information Retrieval."

## 📋 Overview

This experiment demonstrates:
1. **SVD Analysis**: Decomposes a trained bilinear matrix W* into its singular values
2. **Low-rank Approximation**: Tests various rank-r approximations W_r
3. **Performance vs. Rank**: Shows how retrieval performance relates to approximation rank
4. **Error Bound Validation**: Verifies theoretical pointwise error bounds from Theorem 4.1

## ⚙️ Prerequisites

- **Experiment 2 completed**: You need a trained full-rank or high-rank bilinear model
- **MS MARCO data**: Same setup as Experiment 2 (embeddings, dev data, etc.)

## 🚀 Quick Start

```bash
cd experiment3/
python -m experiment3.main_experiment3
```

**Time Required:** ~30-60 minutes (depending on number of ranks tested)

## 📁 Directory Structure

```
experiment3/
├── config.py                  # Configuration for Experiment 3
├── main_experiment3.py        # Main analysis script
├── __init__.py               # Package structure
└── saved_results_exp3/       # Results directory (created)
    ├── experiment3.log       # Detailed logs
    ├── experiment3_summary.json  # Summary results
    ├── singular_values_spectrum.png  # Singular value plot
    ├── performance_vs_rank.png  # Performance vs rank plot
    ├── performance_vs_sigma_r_plus_1.png  # Performance vs σ_{r+1}
    └── pointwise_error_verification.csv  # Error bound verification
```

## ⚙️ Configuration

### Key Settings in `config.py`

```python
# Path to trained model from Experiment 2
PRETRAINED_W_STAR_MODEL_PATH = "experiment2/saved_models/full_rank_bilinear/best_model.pth"
PRETRAINED_W_STAR_MODEL_KEY = "full_rank_bilinear"

# Ranks to test for approximation
EXP3_RANKS_TO_TEST = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

# Pointwise error verification
VERIFY_POINTWISE_ERROR_BOUND = True
NUM_POINTWISE_ERROR_SAMPLES = 10000
```

### Critical Configuration Steps

1. **Update Model Path**: Set `PRETRAINED_W_STAR_MODEL_PATH` to your trained model from Experiment 2
2. **Verify Model Key**: Ensure `PRETRAINED_W_STAR_MODEL_KEY` matches the key used in experiment2/config.py
3. **Choose Ranks**: Adjust `EXP3_RANKS_TO_TEST` based on your needs (smaller list = faster execution)

## 🏃‍♂️ Running the Experiment

### Step 1: Ensure Experiment 2 is Complete

```bash
# Verify you have a trained model
ls experiment2/saved_models/full_rank_bilinear/
# Should show: best_model.pth (or similar)
```

### Step 2: Configure Experiment 3

Edit `experiment3/config.py`:
```python
# Point to your trained model
PRETRAINED_W_STAR_MODEL_PATH = "experiment2/saved_models/full_rank_bilinear/best_model.pth"
PRETRAINED_W_STAR_MODEL_KEY = "full_rank_bilinear"
```

### Step 3: Run the Analysis

```bash
# From project root directory
python -m experiment3.main_experiment3
```

The script will:
1. Load your trained bilinear model
2. Extract the W* matrix (full-rank or reconstruct from low-rank)
3. Perform SVD decomposition
4. Test various rank-r approximations
5. Evaluate each approximation on MS MARCO dev set
6. Generate plots and analysis

## 📊 Understanding the Results

### Expected Outputs

1. **Singular Value Spectrum**: Shows how singular values decay
   - Steep drop indicates most information is in top ranks
   - Flat tail suggests diminishing returns for higher ranks

2. **Performance vs. Rank**: Shows MRR@10 for different ranks
   - Should increase with rank then plateau
   - Identifies the "sweet spot" rank for efficiency

3. **Performance vs. σ_{r+1}**: Relates performance to neglected singular values
   - Validates Theorem 4.1 relationship
   - Smaller σ_{r+1} should correlate with better performance

### Sample Results

```
📊 Key Findings:
- Rank 64 achieves 95% of full-rank performance
- 90% performance reached at rank 32
- Pointwise error bound holds for 99.8% of tested samples
- Singular values decay exponentially after rank 16
```

## 🔬 Technical Details

### What the Experiment Does

1. **SVD Decomposition**: `W* = U @ S @ V^T`
2. **Low-rank Approximation**: `W_r = U_r @ S_r @ V_r^T`
3. **Performance Evaluation**: Tests each W_r on MS MARCO
4. **Error Verification**: Checks `|s_W*(q,d) - s_Wr(q,d)| ≤ σ_{r+1} ||q||_2 ||d||_2`

### Theoretical Validation

- **Theorem 4.1**: Low-rank approximation error is bounded by neglected singular values
- **Practice**: Shows how many ranks needed for good performance
- **Efficiency**: Identifies optimal rank for speed/accuracy trade-off

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```
Error: Model file not found
Solution: Check PRETRAINED_W_STAR_MODEL_PATH in config.py
```

**2. Memory Issues**
```
CUDA out of memory during SVD
Solution: SVD is performed on CPU, but if still issues:
- Reduce NUM_POINTWISE_ERROR_SAMPLES
- Use smaller embedding dimension model
```

**3. Missing Data Files**
```
Error loading embeddings
Solution: Ensure Experiment 2 was completed successfully
```

### Debugging Tips

1. **Check model types**: Verify your model is FullRankBilinearModel or LowRankBilinearModel
2. **Start small**: Test with fewer ranks first
3. **Check logs**: Look at `experiment3.log` for detailed error messages

## 📈 Visualization Examples

All plots are automatically generated and saved to the results directory:

```python
# Example: Loading and analyzing results
import json
import matplotlib.pyplot as plt

# Load results
with open('saved_results_exp3/experiment3_summary.json', 'r') as f:
    results = json.load(f)

# Plot efficiency frontier
ranks = results['ranks_evaluated']
scores = results['mrr_scores']
plt.plot(ranks, scores, 'o-')
plt.xlabel('Rank')
plt.ylabel('MRR@10')
plt.title('Performance vs. Approximation Rank')
plt.show()
```

## 🎯 Key Insights

This experiment typically reveals:

1. **Efficiency Gains**: Most performance achieved with much lower rank than full matrix
2. **Diminishing Returns**: Performance plateaus beyond certain rank
3. **Theoretical Validation**: Error bounds from Theorem 4.1 hold in practice
4. **Practical Guidelines**: Optimal rank selection for deployment

## ✅ Success Criteria

The experiment is successful if:

- [ ] SVD completes successfully
- [ ] Multiple ranks are evaluated
- [ ] Performance increases with rank initially, then plateaus
- [ ] Pointwise error bounds hold (>95% of cases)
- [ ] All plots generate correctly
- [ ] Clear "elbow" visible in performance vs. rank curve

## 📚 Related Resources

- **Theorem 4.1**: See main paper for theoretical foundations
- **Experiment 2**: See `../README_experiment2.md` for model training
- **Full Project**: See `../README.md` for complete overview

## 🔄 Next Steps

After completing this experiment:

1. **Analyze the plots**: Identify optimal rank for your use case
2. **Compare efficiency**: Note parameter reduction vs. performance loss
3. **Update Experiment 2**: Consider using optimal rank in production models
4. **Write conclusions**: Document findings for your paper/report

---

**Note**: This experiment requires a successfully trained model from Experiment 2. The quality of results depends on the original model's performance.
