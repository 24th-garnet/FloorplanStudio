# Floorplan Reference Index

最終更新: 2026-06-17
目的: このプロジェクトの「平面図らしい表現」実装に必要な一次情報・実務情報を再参照できるようにする。

## 1) OpenUSD / USDZ (公式仕様)

- OpenUSD - Usdz File Format Specification  
  <https://openusd.org/dev/spec_usdz.html>
- OpenUSD - Specifications Index  
  <https://openusd.org/dev/spec.html>
- OpenUSD - UsdGeomXformOp Class Reference  
  <https://openusd.org/dev/api/class_usd_geom_xform_op.html>
- NVIDIA Learn OpenUSD - XformCommonAPI  
  <https://docs.nvidia.com/learn-openusd/latest/scene-description-blueprints/xformcommonapi.html>

### 実装で見るポイント

- `xformOp:transform` の解釈
- 回転順序とクォータニオンの扱い
- `usdz` パッケージの制約（非圧縮 ZIP など）

## 2) 建築製図ルール（国際 / 実務）

- ISO 128-23: Lines on construction drawings  
  <https://www.iso.org/standard/22292.html>
- National CAD Standard (NCS) v6  
  <https://www.nationalcadstandard.org/ncs6/>
- NCS v6 Plotting Guidelines (lineweight 参考)  
  <https://www.nationalcadstandard.org/ncs6/pdfs/ncs6_pg.pdf>
- Revit Help - Line Weights  
  <https://help.autodesk.com/cloudhelp/2024/ENU/Revit-Customize/files/GUID-35EF0EF0-9E1B-42E9-AB53-FC94F9AD8C97.htm>
- Revit Help - Visibility/Graphics  
  <https://help.autodesk.com/cloudhelp/2017/ENU/RevitLT-DocumentPresent/files/GUID-A2FC119B-51D7-4C2E-84ED-CD51983EC532.htm>

### 実装で見るポイント

- 線の階層（壁、家具、補助線）
- 表示優先順位
- 縮尺に応じた線幅運用

## 3) BIMセマンティクス（要素分類）

- buildingSMART IFC - IfcWall  
  <https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/IfcWall.htm>
- buildingSMART IFC - IfcDoor  
  <https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/IfcDoor.htm>
- buildingSMART IFC - IfcWindow  
  <https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/IfcWindow.htm>

### 実装で見るポイント

- 壁・開口・建具の概念分離
- ドアと開口の関係（充填関係）
- 記号化ルールへのマッピング

## 4) JIS文脈（日本）

- JIS は有償本文が多く、Web解説は二次情報になりやすい。
- 最終的な仕様確定には、公式規格本文を必ず参照する。

## 5) 本リポジトリの家具記号（確定仕様）

- **`docs/research/furniture-symbols-v1.md`** — 平面図家具記号（`table` 確定、`chair` / `storage` は未確定）

## 6) DXF 壁抽出 ML PoC（計画）

- **`docs/research/floorplan-ml-poc.md`** — Yytsi/floorplan-to-3d UNet の ONNX 化、DXF ラスタ比較、デフォルト仕様・成功条件・ローカル推論

### 外部参照

- Yytsi/floorplan-to-3d — <https://github.com/Yytsi/floorplan-to-3d>
- Hugging Face 重み — <https://huggingface.co/Yytsi/floorplan-to-3d-walls>

---

## 7) このリポジトリ向けの推奨参照順

1. OpenUSD `xform` 仕様確認
2. NCS / Revit の線幅・表示ルール抽出
3. IFCでカテゴリ定義の粒度を合わせる
4. JISで国内運用に合わせる
