# Floorplan Studio

Flask ベースの Web アプリです。RoomPlan / Scaniverse の USDZ と PolyCAM の DXF を取り込み、3D 編集・平面図表示・壁押し出し・Walkthrough・DXF エクスポートまでを **1 つの Studio UI**（`suite.html`）で扱います。

リポジトリ: [24th-garnet/FloorplanStudio](https://github.com/24th-garnet/FloorplanStudio)

## ルーティング

| パス | 内容 |
|------|------|
| `/` | Studio UI（`templates/suite.html`） |

## 主な機能（Studio サイドバー）

- **Import** — RoomPlan (`.usdz`) / PolyCAM (`.dxf`) / Scaniverse (`.usdz`)
- **Wall Extrusion** — DXF 壁の押し出し（PolyCAM）
- **Objects** — レイヤー・オブジェクト一覧（RoomPlan）
- **Replace** — タグ一括置換・テクスチャ適用など（RoomPlan）
- **Transform** — Move / Transform / Rotate、Undo（RoomPlan）
- **Walkthrough** — 1 人称歩行（Scaniverse）
- **Export** — 編集結果を **DXF** でダウンロード（RoomPlan）

付随 UI:

- 中央 3D ビュー（PROXY / USDZ オーバーレイ）
- Floor Plan ポップアップ（真上投影）

## セットアップ

```bash
cd FloorplanStudio
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

ML 壁検出 PoC（任意）:

```bash
pip install -r scripts/floorplan_ml_poc/requirements-poc.txt
python scripts/floorplan_ml_poc/setup_compare_models.py
```

ONNX の大きな補助ファイル（`*.onnx.data`）はリポジトリに含めていません。必要なら上記スクリプトで再生成してください。

## 起動（ローカル）

```bash
python3 app.py
```

既定では `http://127.0.0.1:8100` で待ち受けます。

環境変数:

| 変数 | 既定 | 説明 |
|------|------|------|
| `HOST` | `0.0.0.0` | バインドアドレス |
| `PORT` | `8100` | ポート |
| `FLASK_DEBUG` | （空） | `1` でデバッグモード |

## Render へのデプロイ

`render.yaml` 付きです。

1. [Render](https://render.com) → **New** → **Blueprint**
2. GitHub の `24th-garnet/FloorplanStudio` を接続
3. 検出されたサービスを作成

もしくは Web Service を手動作成する場合:

- **Build**: `pip install -r requirements.txt`
- **Start**: `gunicorn app:app --bind 0.0.0.0:$PORT`

## ディレクトリ構成（主要）

```
FloorplanStudio/
├── app.py                 # Flask アプリ
├── requirements.txt
├── render.yaml            # Render Blueprint
├── roomplan_export.py     # エクスポート（UI は DXF のみ表示）
├── dxf_parser.py
├── wall_extract.py
├── wall_extrude.py
├── templates/
│   └── suite.html         # UI
├── static/
│   ├── main.js            # Studio 本体
│   ├── suite.js / suite.css
│   ├── style.css
│   ├── walk-bridge.js
│   └── input-sources.js
├── assets/replacements/   # オブジェクト置換用アセット
├── texture/               # floor / wall テクスチャ
├── floorplan_ml_poc/      # ML 壁検出 PoC
└── scripts/               # ML setup / USD 変換など
```

ランタイムで生成される `sessions/` と `uploads/` は `.gitignore` 対象です。

## ライセンス / 備考

- 実行には Pixar USD（`usd-core`）や `usd2gltf` / `trimesh` / `ezdxf` など、`requirements.txt` に列挙の依存が必要です。
- DXF エクスポートは USDZ → GLB → DXF の変換経路を使います。環境によっては依存の導入が必要です。
