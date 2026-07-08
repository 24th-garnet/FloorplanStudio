# Floorplan ML PoC — UNet (ONNX) vs 平行ペア壁抽出

最終更新: 2026-06-17  
対象: DXF 壁抽出パイプライン（`wall_extract.py`）の ML 比較検証  
ステータス: **実装済み** — `floorplan_ml_poc/` + CLI + Flask API + UI

---

## 1. 目的

[Yytsi/floorplan-to-3d](https://github.com/Yytsi/floorplan-to-3d) の ResNet-34 UNet を ONNX にエクスポートし、サンプル DXF をラスタ画像化して、現行の **平行ペアヒューリスティック**（`wall_extract.py`）と比較する。

コスト対効果の高い第一歩として、本番 API 統合や押し出し連携は **スコープ外** とする。

---

## 2. PoC 成功条件（デフォルト）

| 項目 | 値 |
|------|-----|
| 対象図面 | Polycam サンプル DXF **1 枚**（`sessions/7770ec3f-a258-46aa-b302-c634d244570d/`） |
| 主目的 | ML 壁マスクが平行ペア結果と比べて「壁らしい」か、誤検出が減るかを確認 |
| 定量指標 | 壁領域の **IoU / Precision / Recall**（ラスタ比較） |
| 定性成果 | 3 列オーバーレイ PNG（元図 / ML 壁 / 平行ペア）+ 差分画像 |
| スコープ外 | Flask API 統合、壁押出、ファインチューニング、商用ライセンス整理 |

**PoC 成功の目安:** 上記成果物が生成され、目視 + IoU で判断できる状態になること。

---

## 3. モデルとライセンス

| 項目 | 値 |
|------|-----|
| リポジトリ | <https://github.com/Yytsi/floorplan-to-3d> |
| 重み（HF） | <https://huggingface.co/Yytsi/floorplan-to-3d-walls> |
| アーキテクチャ | ResNet-34 encoder + UNet decoder |
| 出力クラス | 4 クラス: `floor`, `wall`, `door`, `window` |
| 利用範囲（暫定） | **社内 PoC のみ** — CubiCasa 系データセットは非商用制限があり得るため、製品化前に再確認 |

---

## 4. DXF → 画像（レンダリング仕様）

モデル入力は **512×512 RGB** だが、細線を潰さないため高解像度でレンダしてから letterbox する。

| 項目 | デフォルト |
|------|-----------|
| レンダ解像度 | 長辺 **2048 px**（アスペクト比維持） |
| 背景 | 白 `#FFFFFF` |
| 線色 | 黒 `#000000` |
| 線幅 | **1.5 px**（2048 基準） |
| 余白 | 図面 bbox の **8%** パディング |
| 座標系 | 既存 DXF パイプラインと同様、viewer 平面は **XZ**（`dxf_parser.py` の単位・センタリングに追従） |

### 必須メタデータ

レンダ時に `render_meta.json` を保存し、マスクを DXF / viewer 座標へ逆変換できるようにする。

```json
{
  "render_width_px": 2048,
  "render_height_px": 1536,
  "bbox_xz_m": [xmin, zmin, xmax, zmax],
  "padding_ratio": 0.08,
  "letterbox": {
    "input_size": 512,
    "scale": 0.25,
    "pad_left": 0,
    "pad_top": 64
  },
  "unit_scale_to_meters": 1.0
}
```

---

## 5. モデル前処理（学習と一致）

[floorplan-to-3d](https://github.com/Yytsi/floorplan-to-3d) の `buildingcv` に合わせる。

| 項目 | 仕様 |
|------|------|
| リサイズ | アスペクト比維持の **letterbox** → 512×512 |
| パディング色 | ImageNet 平均色 |
| 正規化 | ImageNet mean / std |
| テンソル形状 | `[1, 3, 512, 512]` float32 |

**方針:** 前処理は PoC ではリポジトリのコードをコピーして固定し、再実装によるズレを避ける。

---

## 6. ONNX エクスポート方針

| 項目 | デフォルト |
|------|-----------|
| ONNX の境界 | **`512×512 tensor → logits` のみ**（前処理・argmax・座標逆変換は Python 側） |
| Opset | 18（torch 2.12 のデフォルトエクスポータ） |
| 入力 | `input`: `[1, 3, 512, 512]` |
| 出力 | `logits`: `[1, 4, 512, 512]` |
| 検証 | PyTorch 出力との max abs diff **< 1e-4** |

成果物の配置（予定）:

```
scripts/floorplan_ml_poc/
  export_onnx.py          # 一度だけ実行（要 torch）
  compare.py              # DXF → 推論 → 比較
  models/
    floorplan-walls.onnx  # エクスポート結果（git 管理は要検討、HF から再生成可）
```

---

## 7. 平行ペア結果との比較定義

ベクタ（平行ペア）とマスク（ML）は形式が異なるため、**ラスタ比較** に統一する。

### 手順

1. ML の `wall` クラスを argmax で取得 → letterbox 逆変換 → レンダ解像度のマスク
2. `wall_extract.json` の各壁を、同解像度の **帯状マスク** に描画（幅 = `thickness_m` をメートル→ピクセル変換）
3. IoU / Precision / Recall を計算
4. 差分画像を出力（例: FP=赤、FN=青）

### 成果物の出力先

```
sessions/<session_id>/ml_poc/
  render.png
  render_meta.json
  ml_wall_mask.png
  heuristic_wall_mask.png
  overlay_compare.png
  diff.png
  compare_report.json
```

`compare_report.json` 例:

```json
{
  "session_id": "7770ec3f-...",
  "wall_iou": 0.62,
  "wall_precision": 0.71,
  "wall_recall": 0.58,
  "heuristic_wall_count": 39,
  "inference_ms_cpu": 180
}
```

---

## 8. ローカルでの ONNX 実行と性能確認

**結論: はい、完全にローカルで動かせます。** クラウド推論は不要です。

### ランタイム

| 用途 | パッケージ | 備考 |
|------|-----------|------|
| 推論（日常） | `onnxruntime` | CPU のみで PoC 十分 |
| GPU 推論（任意） | `onnxruntime-gpu` | CUDA 環境がある場合 |
| エクスポート（一度） | `torch`, `onnx` | 重み取得・変換時のみ |

Flask アプリ（`app.py`）とは独立した **CLI スクリプト** で実行する想定。サーバー再起動やブラウザは不要。

### ローカルで確認できること

1. **数値一致** — 同一入力で PyTorch logits と ONNX logits の差分
2. **推論レイテンシ** — `time.perf_counter()` で 1 枚あたり ms（ウォームアップ後に N 回平均）
3. **メモリ使用量** — プロセス RSS（モデルサイズは通常数十〜百 MB 程度）
4. **視覚品質** — 壁マスク・オーバーレイ PNG の目視
5. **ベースライン比較** — 平行ペアとの IoU / PR

### おおよその性能感（参考）

ResNet-34 UNet・512×512・バッチ 1 の場合:

- **CPU（一般的な開発マシン）:** おおよそ **100〜400 ms / 枚**
- **GPU（CUDA）:** おおよそ **10〜50 ms / 枚**

PoC では図面 1 枚なので CPU で十分。本番でリアルタイム連続推論が必要になった段階で GPU やバッチ化を検討する。

### 最小動作確認（エクスポート後）

```bash
. .venv/bin/activate
pip install onnxruntime numpy opencv-python-headless  # PoC 用（本番 requirements とは別管理可）

python -c "
import numpy as np, onnxruntime as ort, time
sess = ort.InferenceSession('scripts/floorplan_ml_poc/models/floorplan-walls.onnx')
x = np.random.randn(1, 3, 512, 512).astype('float32')
# warmup
sess.run(None, {'input': x})
t0 = time.perf_counter()
for _ in range(10):
    sess.run(None, {'input': x})
print(f'mean_ms={(time.perf_counter()-t0)*100:.1f}')
"
```

### 依存関係の扱い

- **エクスポート用** `torch` は PoC スクリプト実行時のみ（`requirements.txt` には入れない方針）
- **推論用** `onnxruntime` は PoC 確定後に `requirements.txt` へ追加を検討

---

## 9. 実装チェックリスト

```
[x] HF から重み取得 (`export_onnx.py` が自動ダウンロード)
[x] export_onnx.py — PyTorch → ONNX、数値検証
[x] DXF ラスタ化 + render_meta.json (`floorplan_ml_poc/dxf_render.py`)
[x] 前処理（letterbox + ImageNet norm）
[x] ONNX 推論 + wall マスク逆変換
[x] 平行ペア → 帯状マスク描画
[x] IoU / PR 計算 + PNG 出力
[x] compare_report.json
[x] Flask `POST/GET /api/ml-poc/<session_id>` + UI「ML壁比較 (PoC)」
```

### 使い方

```bash
# 1) 推論用（requirements.txt に onnxruntime, pillow 済み）
. .venv/bin/activate
pip install -r scripts/floorplan_ml_poc/requirements-poc.txt   # 初回のみ（ONNX エクスポート用 torch 含む）
python scripts/floorplan_ml_poc/export_onnx.py                 # 初回のみ（~100MB ONNX 生成）

# 2) CLI で比較
python scripts/floorplan_ml_poc/compare_cli.py <session_id>

# 3) Web UI — DXF インポート後「ML壁比較 (PoC)」ボタン
HOST=127.0.0.1 PORT=8100 python app.py
```

---

## 10. 見送り条件（早期判断用）

| シグナル | 解釈 |
|----------|------|
| Polycam DXF で壁がほとんど検出されない | ドメインギャップ大 → ファインチューニング or 別モデル |
| ML は良いが IoU が極端に低い | 座標変換 or 比較定義のバグを先に疑う |
| ONNX と PyTorch で結果がずれる | 前処理不一致 — export 入力を固定 tensor で再検証 |

---

## 11. 関連ファイル（現行パイプライン）

| 役割 | パス |
|------|------|
| 平行ペア壁抽出 | `wall_extract.py` |
| 壁押出（PoC 対象外） | `wall_extrude.py` |
| DXF パース | `dxf_parser.py` |
| 抽出 API | `POST /api/extract-walls/<session_id>` |
| サンプルセッション | `sessions/7770ec3f-a258-46aa-b302-c634d244570d/` |

---

## 12. 次のステップ

1. `scripts/floorplan_ml_poc/` スキャフォールド作成
2. `export_onnx.py` で ONNX 生成・ローカル推論ベンチ
3. `compare.py` でサンプル DXF 1 枚を end-to-end 実行
4. 結果を `sessions/.../ml_poc/` に保存し、成功条件で判断

## 13. DXF ラスタ vs DXF→SVG→cairosvg 比較（2026-06-17 実装）

**CLI:** `python scripts/floorplan_ml_poc/compare_renders_cli.py <session_id>`

| 経路 | 説明 |
|------|------|
| `dxf_pil` | 現行 PoC（PIL で直接ラスタ、白背景） |
| `dxf_svg_cairosvg` | 同一ジオメトリを SVG 化 → cairosvg（ImageNet 平均背景、floorplan-to-3d 準拠） |

成果物: `sessions/<id>/ml_poc/render_compare/`（5 列 `overlay_compare.png` 含む）

**サンプル結果（Polycam DXF）:**

| 指標 | dxf_pil | dxf_svg_cairosvg |
|------|---------|------------------|
| vs 平行ペア IoU | 0.008 | **0.37** |
| Recall | 0.008 | **0.42** |
| ML 壁ピクセル | 1,264 | 51,340 |

→ **レンダラ差だけで ML は大きく改善**するが、平行ペア（89k px）にはまだ届かない。
