# histo-wcsp

AI methods for histopathology: labeling tissue regions (tumor, stroma, healthy, ...)
using Weighted Constraint Satisfaction Problems (WCSP) and related combinatorial
optimization techniques.

## Layout

- `sample_problems/` - warm-up problems cast as WCSPs (domain sizes, arities,
  treewidth) and solved with Toulbar2, plus notes on Local Search / A* alternatives.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytoulbar2
```
