# Floorplan Studio

USDZ 編集・平面図・DXF 壁押出を統合した Web アプリケーションです。

## 含まれる機能

| モジュール | パス | 由来 |
|-----------|------|------|
| ホーム | `/` | 統合ハブ |
| フルエディタ | `/editor` | structureExtractor / topViewer / xdf_extrusion |
| シンプルエディタ | `/simple` | structureExtractor |

### フルエディタの主な機能

- RoomPlan USDZ のインポート・編集・再パック・ダウンロード
- 3D ビュー + 真上投影の平面図
- DXF インポート、壁抽出・押出、ML PoC 比較
- レイヤー管理、オブジェクト置換、テクスチャ適用

## セットアップ

```bash
cd floorplan-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

オプション（ML PoC 用）:

```bash
pip install -r scripts/floorplan_ml_poc/requirements-poc.txt
python scripts/floorplan_ml_poc/setup_compare_models.py
```

## 起動

```bash
python3 app.py
```

ブラウザで `http://localhost:8100` を開き、ホームから各モジュールへ移動します。

環境変数:

- `HOST` — バインドアドレス（既定: `0.0.0.0`）
- `PORT` — ポート（既定: `8100`）
- `FLASK_DEBUG` — `1` でデバッグモード

## ディレクトリ構成

```
floorplan-studio/
├── app.py                 # Flask 統合サーバ
├── templates/
│   ├── hub.html           # ホーム
│   ├── editor.html        # フルエディタ
│   ├── simple_editor.html # シンプルエディタ
├── static/
│   ├── main.js            # フルエディタ UI
│   ├── simple/            # シンプルエディタ UI
├── floorplan_ml_poc/      # ML 壁検出 PoC
├── assets/replacements/   # 置換用 USDZ
└── texture/               # テクスチャ素材
```

## GitHub への追跡

このディレクトリを単独リポジトリとして GitHub に登録する場合:

```bash
cd floorplan-studio
git init
git add .
git commit -m "Initial unified Floorplan Studio application"
git remote add origin <your-repo-url>
git push -u origin main
```

`sessions/` と `uploads/` は `.gitignore` で除外されています。
