# Ester Oil Molecular Design

## Installation

Requires **Python 3.11**.

```bash
git clone https://github.com/chenshuohang/ester-oil-design.git
cd ester-oil-design

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -r environments/requirements-qsar.txt
python -m pip install -e . --no-deps
```

For molecular generation, also install:

```bash
python -m pip install -r environments/requirements-generation.txt
```
