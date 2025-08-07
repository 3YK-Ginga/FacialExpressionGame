# 表情筋トレーニングゲーム (人工知能応用演習課題)

## インストール手順

1. リポジトリをクローン
```bash
git clone https://github.com/3YK-Ginga/FacialExpressionGame.git
cd FacialExpressionGame
```

2. [<u>こちら</u>](https://drive.google.com/file/d/1vYK655C9rNwyfsVHFJAkVxJpms9KsKj_/view?usp=sharing)から表情認識モデルをダウンロードしてモデルを配置
```bash
models/
  best_resemotenet_model.pth
  yolov8n-face-lindevs.pt
```

3. 環境構築(Windows)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 実行
```bash
python game.py
```