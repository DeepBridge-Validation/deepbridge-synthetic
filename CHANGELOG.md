# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-alpha.1] - 2026-02-16

### Added

**Initial release as a standalone package!**

This package was extracted from DeepBridge v1.x to provide focused synthetic data generation capabilities.

**IMPORTANT:** This is a **standalone package** with **no dependencies on DeepBridge core**. It can be used independently for synthetic data generation.

#### Features
- **Gaussian Copula Method**: Statistical modeling for realistic synthetic data generation
- **Privacy-Preserving**: Generate data while protecting individual privacy
- **Quality Metrics**: Evaluate synthetic data quality and fidelity
- **Distributed Computing**: Uses Dask for handling large datasets
- **Multiple Generation Methods**: Various algorithms for different use cases
- **Synthesize API**: Simple, unified interface for all generation methods
- **Comprehensive Testing**: Full test suite with >70% coverage

#### Examples
- `gaussian_copula_example.py`: Complete end-to-end synthetic data generation example

#### Documentation
- Comprehensive README with quick start guide
- Migration guide from DeepBridge v1.x
- API documentation
- Examples directory

### Changed
- **Package name**: `deepbridge.synthetic` → `deepbridge_synthetic`
- **Import path**: Updated to use new package structure
- **Dependencies**: Now completely standalone (no deepbridge dependency!)
- **Structure**: Reorganized as standalone package with own CI/CD

### Migration from DeepBridge v1.x

If you were using `deepbridge.synthetic` in v1.x:

**Before (v1.x):**
```python
from deepbridge.synthetic import Synthesize
```

**After (v2.0):**
```python
from deepbridge_synthetic import Synthesize
```

**Installation:**
```bash
# v1.x
pip install deepbridge  # Includes everything

# v2.0
pip install deepbridge-synthetic  # Standalone!
```

**Key Difference:** Unlike `deepbridge-distillation`, this package is **completely independent** and can be used without installing DeepBridge core.

See the full [Migration Guide](https://github.com/DeepBridge-Validation/DeepBridge/blob/feat/split-repos-v2/desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md) for details.

---

## Use Cases

### Standalone Usage
```python
# Just synthetic data generation - no DeepBridge needed!
pip install deepbridge-synthetic

from deepbridge_synthetic import Synthesize

synthesizer = Synthesize(data=df, method='gaussian_copula')
synthetic_df = synthesizer.generate(n_samples=10000)
```

### With DeepBridge Validation
```python
# If you want both synthetic data AND validation
pip install deepbridge deepbridge-synthetic

from deepbridge_synthetic import Synthesize
from deepbridge import DBDataset, Experiment

# Generate synthetic data
synthetic_df = Synthesize(data=df).generate(n_samples=1000)

# Validate model on synthetic data
dataset = DBDataset(data=synthetic_df, target_column='target')
experiment = Experiment(dataset=dataset, models={'model': model})
```

---

## Related Projects

- **[deepbridge](https://github.com/DeepBridge-Validation/deepbridge)** - Model Validation Toolkit (core)
  - NOT a dependency - use if you need model validation

- **[deepbridge-distillation](https://github.com/DeepBridge-Validation/deepbridge-distillation)** - Model Distillation
  - Model compression and knowledge distillation
  - Requires deepbridge as dependency

---

## Support

- **Issues**: [GitHub Issues](https://github.com/DeepBridge-Validation/deepbridge-synthetic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DeepBridge-Validation/deepbridge-synthetic/discussions)
- **Documentation**: https://deepbridge.readthedocs.io/en/latest/synthetic/
- **Email**: gustavo.haase@gmail.com

---

**Maintainers**: Gustavo Haase, Paulo Dourado
**License**: MIT
