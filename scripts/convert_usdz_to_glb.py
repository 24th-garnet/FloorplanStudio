import argparse
from pathlib import Path
import sys

import bpy


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def convert_one(src: Path, dst: Path) -> None:
    clear_scene()
    bpy.ops.wm.usd_import(filepath=str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(dst),
        export_format="GLB",
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for name in args.files:
        src = input_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing source: {src}")
        dst = output_dir / f"{src.stem}.glb"
        print(f"[convert] {src} -> {dst}")
        convert_one(src, dst)


if __name__ == "__main__":
    main()
