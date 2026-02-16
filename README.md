# DeepBridge Synthetic

Privacy-Preserving Synthetic Data Generation

**Note:** This is a standalone library and can be used without installing DeepBridge.

## Installation

```bash
pip install deepbridge-synthetic
```

## Quick Start

```python
from deepbridge_synthetic import Synthesize

# Generate synthetic data
synthesizer = Synthesize(
    data=original_df,
    method='gaussian_copula'
)

synthetic_df = synthesizer.generate(n_samples=10000)
```

## Features

- **Gaussian Copula**: Statistical modeling for synthetic data
- **Privacy-Preserving**: Generate data while protecting privacy
- **Quality Metrics**: Evaluate synthetic data quality
- **Distributed Computing**: Uses Dask for large datasets
- **Multiple Methods**: Various generation algorithms

## Documentation

Full documentation: https://deepbridge.readthedocs.io/en/latest/synthetic/

## Related Projects

- [deepbridge](https://github.com/DeepBridge-Validation/deepbridge) - Model Validation Toolkit
- [deepbridge-distillation](https://github.com/DeepBridge-Validation/deepbridge-distillation) - Model Distillation

## License

MIT License - see [LICENSE](LICENSE)
