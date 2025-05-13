# Experiment 1: Structured Agreement Task - Synthetic Validation

This experiment directly validates **Theorem 3.1** from "On the Theoretical Advantages of Bilinear Similarities in Dense Information Retrieval" using synthetic data and the Structured Agreement Ranking Task.

## 📋 Overview

This experiment demonstrates:
1. **Bilinear Sufficiency (Theorem 3.1.i)**: Rank-2 bilinear models achieve perfect performance on the agreement task
2. **WDP Universality Failure (Theorem 3.1.ii)**: No single weighted dot-product model can solve all cases
3. **Concrete separation**: Clear evidence where feature interactions are essential for relevance

## 🚀 Quick Start

```bash
cd experiment1/
python experiment1.py
```

**Time Required:** ~5-10 minutes  
**Expected Results:** Bilinear 100% success, WDP <60% success

## 📁 Directory Structure

```
experiment1/
├── config.py              # Configuration parameters
├── experiment1.py         # Main experiment script  
├── models.py              # Bilinear and WDP model implementations
├── synthetic_data_gen.py  # Data generation utilities
├── __init__.py           # Empty file for package structure
└── saved_models_exp1/    # Results will be saved here (created)
```

## ⚙️ Configuration Parameters

All experiment parameters are controlled through `config.py`:

### Core Experiment Parameters

```python
# Dimension of hypercube vectors {-1, +1}^n
DIM_N = 10
```
- **Default**: 10  
- **Requirements**: Must be ≥ 3 for WDP failure demonstration
- **Impact**: Higher dimensions make the problem harder for WDP but don't affect bilinear

```python
# Number of random (q, I0) pairs to test for bilinear model
NUM_TEST_CASES_BILINEAR = 1000
```
- **Default**: 1000
- **Purpose**: Tests bilinear sufficiency across different queries and I0 sets
- **Expected**: Should give 100% success rate

```python
# Number of (q, I0) pairs to test WDP generalization  
NUM_TEST_CASES_WDP_GENERALIZATION = 1000
```
- **Default**: 1000
- **Purpose**: Tests how well a single WDP model generalizes across different I0 sets
- **Expected**: Should show variable performance

### Advanced Parameters

```python
# WDP Training Parameters
WDP_TRAIN_SAMPLES = 50000    # Training samples for WDP
WDP_LEARNING_RATE = 1e-3     # Learning rate for AdamW
WDP_EPOCHS = 5               # Training epochs
WDP_BATCH_SIZE = 128         # Batch size for training
WDP_MARGIN = 1.0             # Margin for ranking loss

# Analysis Control Flags
SAVE_DETAILED_RESULTS = True          # Save results for plotting
ANALYZE_FAILURE_PATTERNS = True       # Analyze specific failure cases  
INCLUDE_SPECIALIZED_WDP_TEST = True    # Test specialized WDP models
```

## 🏃‍♂️ Running the Experiment

### Step 1: Environment Setup

```bash
# Create environment (optional but recommended)
conda create -n bilinear_exp python=3.9
conda activate bilinear_exp

# Install dependencies
pip install torch numpy scipy pandas tqdm
```

### Step 2: Quick Configuration

For a test run, edit `config.py`:

```python
# Quick test settings
DIM_N = 6
NUM_TEST_CASES_BILINEAR = 100
NUM_TEST_CASES_WDP_GENERALIZATION = 100
WDP_TRAIN_SAMPLES = 5000
```

For thorough validation:
```python  
# Thorough settings
DIM_N = 10
NUM_TEST_CASES_BILINEAR = 2000
NUM_TEST_CASES_WDP_GENERALIZATION = 2000
WDP_TRAIN_SAMPLES = 50000
```

### Step 3: Run the Experiment

```bash
python experiment1.py
```

The script will show:
- Progress bars for each test phase
- Real-time loss during WDP training  
- Success rates for each component
- Detailed analysis of failure patterns

### Step 4: Check Results

Results are saved in `saved_models_exp1/Exp1_StructuredAgreement_n{DIM_N}_{timestamp}/`:

```
saved_models_exp1/
└── Exp1_StructuredAgreement_n10_20241213-143022/
    ├── experiment.log                    # Detailed log file
    ├── experiment1_results.json         # Main results summary
    ├── wdp_performance_by_I0.csv       # Performance by I0 set
    ├── score_distributions.json         # Score distributions
    ├── specialized_wdp_results.json     # Specialized WDP results
    └── failure_analysis.csv            # Failure pattern analysis
```

## 📊 Understanding the Results  

### Expected Outcomes

1. **Bilinear Success Rate**: Should be exactly **100.0%**
   - Validates Theorem 3.1.i: Bilinear models can solve the task perfectly

2. **WDP Success Rate**: Should be significantly lower (typically **20-60%**)
   - Validates Theorem 3.1.ii: No single WDP can solve all cases

3. **Performance Variability**: WDP performance should vary across different I0 sets
   - Demonstrates the universality failure

### Sample Output

```
📊 Key Results:
  ✓ Bilinear Success Rate: 100.00% (95% CI: [99.63%, 100.00%])
  ✗ WDP Success Rate: 45.60% (95% CI: [43.15%, 48.05%])

📈 WDP Performance Variability:
  - Performance varies from 15.2% to 78.9% across different I0 sets
  - Standard deviation: 18.7%
  - This variability demonstrates the universality failure

🔬 Theoretical Validation:
  ✓ Theorem 3.1.i: Bilinear models achieve perfect performance
  ✓ Theorem 3.1.ii: No single WDP can universally solve the task
```

## 🔬 Technical Details

### The Structured Agreement Task

Given:
- Query `q ∈ {-1, +1}^n`
- Index set `I0 = {i1, i2}` (two dimensions)
- Document set `D(q, I0) = {q, q⊙e_I0, -q, -(q⊙e_I0)}`

**Task**: Rank documents that agree with `q` on `I0` above those that disagree.

### Why Bilinear Succeeds

The theoretical solution uses `W_I0 = e_i1*e_i1^T + e_i2*e_i2^T`:
- Score for agreeing docs: `+2`
- Score for disagreeing docs: `-2`  
- Perfect separation achieved

### Why WDP Fails

WDP uses fixed weights `v` that must work for all possible `I0` sets:
- Different `I0` sets require prioritizing different dimensions
- No single weight vector can satisfy all conflicting requirements
- Results in inconsistent performance across `I0` sets

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# In config.py, reduce batch size:
WDP_BATCH_SIZE = 64  # or 32

# Or disable CUDA:
DEVICE = "cpu"
```

**2. Slow Performance**  
```python
# Reduce test cases for faster runs:
NUM_TEST_CASES_BILINEAR = 100
NUM_TEST_CASES_WDP_GENERALIZATION = 100
```

**3. Import Errors**
```bash
# Ensure you're in the correct directory
cd experiment1/  
python experiment1.py
```

### Validation Checks

The experiment includes several assertions:
1. Bilinear success rate must be 100% (within 1e-3 tolerance)
2. WDP failure cases are logged for debugging
3. Confidence intervals are computed for statistical validation

## 🔬 Advanced Usage

### Running Specific Parts

You can modify `main()` in `experiment1.py` to run only certain parts:

```python
def main():
    # ... setup code ...
    
    # Run only bilinear test
    results_bilinear = test_bilinear_sufficiency(logger)
    
    # Skip WDP testing
    # results_wdp = test_wdp_universality_failure(trained_wdp, logger)
```

### Adding Custom Analysis

To add custom analysis patterns, edit `synthetic_data_gen.py`:

```python
def generate_challenging_patterns(n_dim):
    patterns = [
        # ... existing patterns ...
        
        # Add your custom pattern:
        {
            "q": torch.tensor([1, -1, 1, -1] + [-1] * (n_dim - 4), dtype=torch.float32),
            "I0": (0, 3),
            "name": "Custom alternating pattern",
            "description": "Tests alternating query pattern"
        }
    ]
    return patterns
```

### Visualization Examples

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load WDP performance by I0
df = pd.read_csv('saved_models_exp1/.../wdp_performance_by_I0.csv')

# Plot performance variability
plt.figure(figsize=(10, 6))
plt.bar(range(len(df)), df['success_rate'], alpha=0.7)
plt.xlabel('I0 Set Index')
plt.ylabel('Success Rate (%)')
plt.title('WDP Performance Varies by I0 Set')
plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='Bilinear Performance (100%)')
plt.legend()
plt.show()

# Score distribution analysis
import json
with open('saved_models_exp1/.../score_distributions.json', 'r') as f:
    scores = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bilinear scores (should be exactly +2 and -2)
ax1.hist(scores['agree_scores'], alpha=0.7, label='Agree docs')
ax1.hist(scores['disagree_scores'], alpha=0.7, label='Disagree docs') 
ax1.set_xlabel('Score')
ax1.set_ylabel('Count')
ax1.set_title('Bilinear Model Score Distribution')
ax1.legend()

# WDP score distribution analysis would require additional data
plt.tight_layout()
plt.show()
```

## ✅ Validation Checklist

Before considering the experiment complete:

- [ ] Bilinear success rate is 100% (validates Theorem 3.1.i)
- [ ] WDP success rate is significantly lower and varies by I0
- [ ] Specialized WDP models perform well on their training I0 but fail on others
- [ ] Failure analysis shows challenging patterns where WDP struggles
- [ ] All results are saved with confidence intervals

## 🎯 Key Takeaways

This experiment demonstrates:

1. **Theoretical Validation**: Direct empirical proof of Theorem 3.1
2. **Practical Evidence**: Concrete examples where bilinear models outperform WDP
3. **Feature Interaction Importance**: Shows why modeling interactions matters
4. **Statistical Rigor**: Provides confidence intervals and comprehensive analysis

The results provide strong empirical evidence for the theoretical advantages of bilinear similarities in tasks requiring feature interactions - a core contribution of the paper.

## 📚 Related Resources

- **Main README**: See `../README.md` for project overview
- **Experiment 2**: See `../README_experiment2.md` for real-world validation
- **Paper**: [Link when available]
- **Theory Reference**: Theorem 3.1 in the main paper