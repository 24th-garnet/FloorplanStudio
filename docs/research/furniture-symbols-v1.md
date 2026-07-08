# Furniture Floorplan Symbols v1

最終更新: 2026-05-19  
実装: `static/main.js`（`drawInteriorSymbol` 系）

カテゴリごとの平面図記号仕様。ユーザー確認済みの見た目は **確定（locked）** として扱い、変更時は本書とコードをセットで更新する。

---

## 確定: `table`（テーブル）

**状態**: locked（2026-05-19 承認）

### 見た目

1. オブジェクトの回転矩形（OBB）に合わせた **グレーの長方形**
2. 長方形 **全面** に斜線ハッチ（クリップ内の対角平行線）
3. 幾何中心に **点線の真円**
4. 真円の中心（＝矩形中心）に文字列 **`table`**

### 色・線

| 要素 | 定数 | 値 |
|------|------|-----|
| 塗り | `THEME_2D.tableGrayFill` | `#c5c9d0` |
| 枠・斜線・円 | `THEME_2D.tableGrayStroke` | `#6f7782` |
| ラベル文字 | `THEME_2D.text` | `#111111` |

### 幾何パラメータ

| 項目 | 値 |
|------|-----|
| 矩形スケール（u/v） | `0.96` |
| 斜線間隔 | `0.12 m`（`hatchObbRect`） |
| 斜線線幅 | `0.55 px` |
| 真円半径 | `min(halfU, halfV) × 0.38`（平面メートル、真円） |
| 点線円 | `[5, 4]`、線幅 `1 px` |
| ラベル | `11px system-ui, sans-serif`、中央揃え |

### 描画順（下→上）

1. `drawObbRect`（塗り＋外枠）
2. `hatchObbRect`（全面斜線）
3. 点線円
4. `table` テキスト

### 実装関数

- `drawTableSymbol`
- 共有: `drawObbRect`, `hatchObbRect`, `getResizeBasis`, `planLocalPoint`

---

## 確定: `chair`（椅子）

**状態**: locked（2026-05-19 承認）

### 見た目

1. **座面**: フットプリントに沿った長方形（OBB）
2. **背もたれ**: −v 側（背面）に **薄い長方形** を重ねる
3. フットプリント幾何中心に文字列 **`chair`**

### 色・線

| 要素 | 定数 | 値 |
|------|------|-----|
| 塗り（座面・背） | `categoryFill2d("chair")` | `THEME_2D.chairFill`（`#6b8ec1`） |
| 枠線 | `THEME_2D.symbol` | `#4a5560` |
| ラベル文字 | `THEME_2D.text` | `#111111` |

### 幾何パラメータ

| 項目 | 値 |
|------|-----|
| 幅スケール（u） | `halfU × 0.94` |
| 背もたれ奥行 | `max(halfV × 0.22, 0.04 m)` |
| 座面奥行 | `halfV − 背もたれ奥行`（中心を +v に `backDepth × 0.5` オフセット） |
| 背もたれ中心 | `planLocalPoint(0, −halfV + backDepth)` |
| ラベル | `11px system-ui, sans-serif`、中央揃え |

### 描画順（下→上）

1. 座面 `drawObbRect`
2. 背もたれ `drawObbRect`
3. `chair` テキスト

### 実装関数

- `drawChairSymbol`
- 共有: `drawObbRect`, `getResizeBasis`, `planLocalPoint`

---

## ドラフト: `storage`（収納・クローゼット）

**状態**: draft（2026-05-19）

### 見た目

1. フットプリントに沿った **長方形**（OBB、スケール `0.96`）
2. **長辺方向**に点線の **中線**のみ（吊り棒／クローゼット表現）
3. フットプリント中心に文字列 **`storage`**

### 色・線

| 要素 | 定数 | 値 |
|------|------|-----|
| 塗り | `categoryFill2d("storage")` | `THEME_2D.storageFill`（`#6aa37e`） |
| 枠・中線 | `THEME_2D.symbol` | `#4a5560` |
| ラベル文字 | `THEME_2D.text` | `#111111` |

### 幾何パラメータ

| 項目 | 値 |
|------|-----|
| 中線 | 長辺の 92%、点線 `[6, 4]` |
| ラベル | `11px system-ui, sans-serif`、中央揃え |

### 描画順（下→上）

1. `drawObbRect`
2. 点線中線
3. `storage` テキスト

### 実装関数

- `drawStorageSymbol`
- 共有: `drawObbRect`, `getResizeBasis`, `planLocalPoint`, `drawSegment`

---

## 参照

- `docs/research/implementation-rules-v1.md` — 平面図レンダラー全般
- `docs/research/floorplan-reference-index.md` — 外部資料リンク
