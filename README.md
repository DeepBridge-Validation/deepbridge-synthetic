# DeepBridge Synthetic

[![Tests](https://github.com/DeepBridge-Validation/deepbridge-synthetic/actions/workflows/tests.yml/badge.svg)](https://github.com/DeepBridge-Validation/deepbridge-synthetic/actions)
[![codecov](https://codecov.io/gh/DeepBridge-Validation/deepbridge-synthetic/branch/main/graph/badge.svg)](https://codecov.io/gh/DeepBridge-Validation/deepbridge-synthetic)
[![PyPI version](https://badge.fury.io/py/deepbridge-synthetic.svg)](https://badge.fury.io/py/deepbridge-synthetic)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Privacy-Preserving Synthetic Data Generation

> **Standalone Package - No Dependencies on DeepBridge!**
>
> This package can be used independently for synthetic data generation.
> It was extracted from DeepBridge v1.x to provide a focused, standalone solution.
> See [Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md) if migrating from v1.x.

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
