# Viewer Behavior v1 (確定スナップショット)

最終更新: 2026-05-19  
対象: `static/main.js`, `app.py`, `texture/`, `templates/index.html`  
目的: 2026-05-19 時点で合意・実装済みの挙動を固定し、以降の変更時の基準にする。

---

## 1. 平面図 — 描画レイヤ

`partitionFloorplanLayers()` で `floor` とそれ以外に分割する。

| 順序 | レイヤ | 内容 |
|------|--------|------|
| 1（最背面） | `floor` | 床フットプリント（`THEME_2D.floorFill`） |
| 2 以降 | `above` | 壁・開口・窓・ドア・家具（`chair` / `table` / `storage`）・その他 |
| 最前面 | オーバーレイ | 選択枠、寸法線（floor）、壁寄せハイライト、スケールバー |

**テクスチャは平面図に描画しない。** 床・壁は単色フットプリントのみ。

実装: `drawTopView()` — `floorLayer` を先に `drawFloorplanItem`、続けて `aboveFloorLayer`。

---

## 2. 平面図 — クリック選択（ヒットテスト）

重なり時の優先度: **floor が最低**（家具・壁などが常に優先）。

```text
pickCandidates = [...floorLayer, ...aboveFloorLayer]
hit = reverse(pickCandidates) で先頭ヒット
```

- 配列の**末尾**（`above` 側）から判定するため、`floor` は配列の**先頭**に置く。
- ❌ `[...above, ...floor].reverse()` は floor が先に判定され table が選べなくなる（2026-05-19 に修正済み）。

ヒット形状: 各オブジェクトの `corners`（`getObjectFootprintCorners` の凸包）内。家具記号の OBB 描画と同じ `corners` を共有。

---

## 3. floor / wall — 選択と編集

`isSurfaceTextureOnlyObject()` = category または name が `floor` / `wall`。

| 操作 | floor / wall | その他 |
|------|----------------|--------|
| 選択（一覧・2D・3D） | ✅ | ✅ |
| 移動・回転・リサイズ・Transform | ❌ | ✅ |
| Apply / Delete | ❌ | ✅ |
| テクスチャ UI | ✅ のみ表示 | 置換 UI |

テクスチャ適用目的での選択のみ許可。

---

## 4. テクスチャ — 適用範囲

| 表示 | テクスチャ |
|------|------------|
| **3D プロキシ**（Show editable proxy boxes） | ✅ |
| **2D 平面図** | ❌ |
| **USDZ オーバーレイ**（実スキャン） | ❌ |
| **ダウンロード USDZ** | ❌（`replacement_state.json` の `texture_asset_key` のみ） |

素材ファイル: `texture/floor.png`, `texture/wall.png`  
API: `TEXTURE_ASSET_MAP`（`app.py`）、`/api/texture-assets`（`mtime` 付き）、`/api/apply-texture/<session_id>`。

キャッシュ: クライアントは `mtime` を `?v=` に使用。ファイル差し替え後はページリロード推奨。

---

## 5. テクスチャ — 3D 貼り方（縦横比維持）

`computeAspectPreservingRepeat(faceW, faceH, texW, texH)` + `applySurfaceTexture()`。

- **テクスチャの長辺** → **面の短辺** に合わせる（伸縮なし）。
- 余白は `RepeatWrapping` でタイル繰り返し。
- `BoxGeometry` の 6 面それぞれにクローンした `map` と個別 `repeat` を設定。

---

## 6. 家具記号（平面図）

`docs/research/furniture-symbols-v1.md` を正とする。

- **table**（locked）: グレー矩形・斜線ハッチ・中心点線円・`table` ラベル
- **chair**（locked）: 座面 + 背もたれ・`chair` ラベル
- **storage**（draft）: 矩形・長辺点線中線・`storage` ラベル

---

## 7. 関連コード（クイック参照）

| 関心 | 関数・定数 |
|------|------------|
| レイヤ分割 | `partitionFloorplanLayers` |
| 平面図描画 | `drawTopView`, `drawFloorplanItem` |
| クリック | `onTopViewPointerDown` |
| 表面のみ選択 | `isSurfaceTextureOnlyObject`, `syncTransformUiForSelection` |
| 3D テクスチャ | `applySurfaceTexture`, `computeAspectPreservingRepeat`, `loadThreeTexture` |
| テクスチャ再読込 | `refreshTextureLibrary`, `syncTextureVersionsFromAssets` |

---

## 8. 底面を床に合わせる（メッシュ伸長）

ボタン: **Snap object bottoms to floor (extend mesh)**  
API: `POST /api/snap-bottoms-to-floor/<session_id>`

- **位置（`matrix4d`）は変更しない。** 底面付近の頂点のみ世界座標 +Y 方向に伸ばす。
- 床上面の参照高さ: 全 `floor` オブジェクトのワールド頂点の **最大 Y**。
- `floor` 自体は対象外。それ以外の全オブジェクト（壁・家具等）が対象。
- Undo: 変更前 USDA を `geometry` スタックで復元（`/api/restore-object-geometry`）。

---

## 9. 変更時のチェックリスト

- [ ] `pickCandidates` の順序を変えた場合、floor 上の table が選べるか
- [ ] 平面図にテクスチャを戻す場合は本書 §1・§4 を更新
- [ ] テクスチャ PNG を差し替えたら `mtime` キャッシュバストが効くか
- [ ] floor/wall で Transform が付かないか

---

## 10. 参考

- `docs/research/implementation-rules-v1.md` — 平面図デザインルール（理想仕様）
- `docs/research/furniture-symbols-v1.md` — 家具記号
- `docs/research/floorplan-flowchart.md` — 処理フロー
