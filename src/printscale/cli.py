"""Command line interface.

    printscale weights                     list and fetch checkpoints
    printscale inspect SCAN                ppi table, grain sigma, suggested crop
    printscale bake    SCAN -o out/        candidates on one crop, with a sheet
    printscale run     SCAN -o out/        full pass, model plus finishing
    printscale finish  SCAN -o out/ --skip-sr   retune finishing from the cache

``run`` caches the model output as ``_sr.png`` in the output directory. The
expensive stage runs once; the finishing passes are seconds apiece, so grain
and sharpening can be tuned against a real proof instead of guessed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from printscale import __version__, files, metrics, pipeline
from printscale.finishing import measure_grain
from printscale.models import WEIGHTS, default_cache_dir, ensure_weights

LOG = logging.getLogger("printscale")

_VERDICTS = ("comfortable", "fine", "mural territory", "soft")


def _add_finish_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("finishing")
    group.add_argument(
        "--grain",
        type=float,
        default=0.55,
        help="grain restoration strength, 0 disables (default: %(default)s)",
    )
    group.add_argument("--lc-radius", type=float, default=18.0)
    group.add_argument("--lc-amount", type=float, default=0.16)
    group.add_argument("--sharp-radius", type=float, default=1.1)
    group.add_argument("--sharp-amount", type=float, default=0.42)
    group.add_argument("--sharp-threshold", type=float, default=0.012)
    group.add_argument("--no-despeckle", action="store_true", help="skip dust removal entirely")
    group.add_argument(
        "--detail-limit",
        type=float,
        default=5.0,
        help="dust detector's neighbourhood flatness gate (default: %(default)s)",
    )
    group.add_argument(
        "--width-mm",
        type=float,
        default=800.0,
        help="intended print width, stamped as DPI (default: %(default)s)",
    )


def _params(args: argparse.Namespace) -> pipeline.FinishParams:
    return pipeline.FinishParams(
        grain=args.grain,
        lc_radius=args.lc_radius,
        lc_amount=args.lc_amount,
        sharp_radius=args.sharp_radius,
        sharp_amount=args.sharp_amount,
        sharp_threshold=args.sharp_threshold,
        despeckle=not args.no_despeckle,
        detail_limit=args.detail_limit,
    )


def _print_ppi(pixel_width: int) -> None:
    print(f"\n  {'print width':>12}  {'effective':>10}   reading")
    for width_mm, ppi, verdict in metrics.ppi_table(pixel_width):
        print(f"  {width_mm:>9.0f} mm  {ppi:>6.0f} dpi   {_VERDICTS[verdict]}")


def cmd_weights(args: argparse.Namespace) -> int:
    cache = args.cache or default_cache_dir()
    if args.fetch:
        for key in args.fetch if args.fetch != ["all"] else list(WEIGHTS):
            ensure_weights(key, cache)
        return 0
    print(f"cache: {cache}\n")
    for key, spec in WEIGHTS.items():
        present = "cached" if (cache / spec.name).is_file() else "      "
        print(f"  {present}  {key:<18} x{spec.scale}  {spec.size / 1e6:6.1f} MB  {spec.note}")
    print("\nfetch with:  printscale weights --fetch x2plus")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    rgb = files.load_rgb(args.source)
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    height, width = gray.shape
    print(f"\n  {width} x {height} px")
    print(f"  grain sigma   {measure_grain(gray):.5f}  ({measure_grain(gray) * 255:.2f}/255)")
    x0, y0, x1, y1 = metrics.detect_crop((gray * 255).astype(np.uint8))
    if (x1 - x0, y1 - y0) != (width - 6, height - 6):
        print(
            f"  photo area    x{x0}..{x1}  y{y0}..{y1}   "
            f"({x1 - x0} x {y1 - y0}) — crop before upscaling"
        )
    _print_ppi(width)
    _print_ppi(width * 2)
    print("\n  (second table is after a 2x pass)\n")
    return 0


def cmd_bake(args: argparse.Namespace) -> int:
    rgb = files.load_rgb(args.source)
    crop = None
    if args.crop:
        parts = [int(p) for p in args.crop.split(",")]
        if len(parts) != 3:
            LOG.error("--crop wants x,y,size — got %r", args.crop)
            return 2
        crop = (parts[0], parts[1], parts[2])

    results = pipeline.bake_off(
        rgb, crop=crop, models=tuple(args.models), threads=args.threads, cache_dir=args.cache
    )
    if not results:
        LOG.error("no candidate produced a result")
        return 1

    full_mp = (rgb.shape[0] * rgb.shape[1]) / 1e6
    print(f"\n  full frame is {full_mp:.2f} MP\n")
    print(f"  {'candidate':<16}{'s/MP':>10}{'full frame':>14}")
    for result in results:
        print(
            f"  {result.name:<16}{result.seconds_per_megapixel:>10.1f}"
            f"{result.projected_minutes(full_mp):>11.1f} min"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / "bakeoff.png"
    cv2.imwrite(str(sheet), pipeline.contact_sheet(results))
    print(f"\n  comparison sheet: {sheet}")
    print("  Look at it. The timings will not tell you about waxy skin.\n")
    return 0


def _run_or_finish(args: argparse.Namespace, *, skip_sr: bool) -> int:
    source = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_png = out_dir / "_sr.png"

    rgb = files.load_rgb(source)
    source_gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    source_gray_f = source_gray.astype(np.float32) / 255.0

    if skip_sr:
        if not cache_png.is_file():
            LOG.error(
                "--skip-sr needs %s, which does not exist. Run `printscale run` first.", cache_png
            )
            return 2
        cached = cv2.imread(str(cache_png), cv2.IMREAD_COLOR)
        if cached is None:
            LOG.error("could not read the cached SR result at %s", cache_png)
            return 2
        upscaled = cv2.cvtColor(cached, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        scale = cached.shape[1] / rgb.shape[1]
        LOG.info("reusing cached %dx%d SR result (x%.2f)", cached.shape[1], cached.shape[0], scale)
    else:
        upscaled, native = pipeline.upscale(
            rgb,
            args.model,
            tile=args.tile,
            overlap=args.overlap,
            threads=args.threads,
            cache_dir=args.cache,
        )
        scale = float(native)
        cv2.imwrite(
            str(cache_png),
            cv2.cvtColor((upscaled * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        LOG.info("cached the model output at %s — rerun with --skip-sr to retune", cache_png)

    gray = pipeline.finish(upscaled, scale, _params(args), source_gray_f)

    height, width = gray.shape
    stem = source.stem
    master = files.save_master(gray, out_dir / f"{stem}_{width}x{height}_16bit.tif", args.width_mm)
    preview = files.save_jpeg(gray, out_dir / f"{stem}_{width}x{height}.jpg", args.width_mm)

    as_uint8 = (gray * 255).round().astype(np.uint8)
    period = int((args.tile - 2 * args.overlap) * scale)
    vertical, horizontal = metrics.seam_ratio(as_uint8, period)
    shadow, highlight = metrics.clipping(as_uint8)
    print("\n  checks")
    print(
        f"    seam ratio      {vertical:.3f} vertical, {horizontal:.3f} horizontal"
        f"   {'ok' if max(vertical, horizontal) < 1.5 else 'SUSPECT — inspect the blend'}"
    )
    print(
        f"    clipping        {shadow * 100:.3f}% shadow, {highlight * 100:.3f}% highlight"
        f"   {'ok' if max(shadow, highlight) < 0.005 else 'high — pull the contrast back'}"
    )
    _print_ppi(width)
    print(f"\n  master   {master}")
    print(f"  preview  {preview}\n")
    print("  Everything above the source Nyquist is synthesised, not recovered.")
    print("  Do not present this as an evidential record.\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return _run_or_finish(args, skip_sr=False)


def cmd_finish(args: argparse.Namespace) -> int:
    return _run_or_finish(args, skip_sr=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="printscale",
        description="Upscale scanned photographs for large-format print.",
    )
    parser.add_argument("--version", action="version", version=f"printscale {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="weights cache directory (default: ~/.cache/printscale/weights)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    weights = subparsers.add_parser("weights", help="list or fetch model checkpoints")
    weights.add_argument(
        "--fetch", nargs="*", metavar="MODEL", help="download and verify these models, or 'all'"
    )
    weights.set_defaults(func=cmd_weights)

    inspect = subparsers.add_parser("inspect", help="measure a source before processing it")
    inspect.add_argument("source", type=Path)
    inspect.set_defaults(func=cmd_inspect)

    bake = subparsers.add_parser("bake", help="compare candidates on one crop")
    bake.add_argument("source", type=Path)
    bake.add_argument("-o", "--out", type=Path, default=Path("out"))
    bake.add_argument("--crop", help="x,y,size — pick the hardest content you have")
    bake.add_argument("--models", nargs="+", default=["x2plus", "x4plus", "general-x4v3"])
    bake.add_argument("--threads", type=int, default=0)
    bake.set_defaults(func=cmd_bake)

    for name, func, help_text in (
        ("run", cmd_run, "full pass: model then finishing"),
        ("finish", cmd_finish, "re-run finishing from the cached model output"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("source", type=Path)
        sub.add_argument("-o", "--out", type=Path, default=Path("out"))
        sub.add_argument("--model", default="x2plus", choices=sorted(WEIGHTS))
        sub.add_argument(
            "--tile",
            type=int,
            default=320,
            help="raise until memory complains (default: %(default)s)",
        )
        sub.add_argument("--overlap", type=int, default=32)
        sub.add_argument(
            "--threads", type=int, default=0, help="torch threads; 0 leaves torch's default alone"
        )
        _add_finish_args(sub)
        sub.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        result: int = args.func(args)
        return result
    except KeyboardInterrupt:
        LOG.warning("interrupted")
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOG.error("%s", exc)
        LOG.debug("traceback", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
