# depth-dpt-test
## Install
```
$ python3 -m venv .venv
$ source .venv/bin/activate.fish
(.venv) $ pip install -r requirements/main.txt
(.venv) $ pip install -r requirements/dev.txt
```

## 学習
See [notebooks/train_DPT_depth.ipynb](./notebooks/train_DPT_depth.ipynb).

## 推論
```
PYTHONPATH=. python scripts/infer.py [SAMPLE_DIR]
```

## 参考サイト
- [simonmeister/pytorch-mono-depth: Monocular depth prediction with PyTorch](https://github.com/simonmeister/pytorch-mono-depth)
- [food-analytic/food-depth-dpt: Fine-tuning Dense Prediction Transformer (DPT) on Nutrition5k for food depth estimation](https://github.com/food-analytic/food-depth-dpt)
