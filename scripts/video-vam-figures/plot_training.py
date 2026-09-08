#!/usr/bin/env python3
"""Render the archived histories; --check validates without writing files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
DATA = HERE / "training-history.json"
OUTPUT = HERE.parents[1] / "static/assets/img/video-vam/cosmos-t16-t2-training.svg"
WIDTH, HEIGHT = 960, 1000
EXPECTED = {
    "ixzworl4": (48, 38000, 13.058368685598639, 6879.7633914150065),
    "jri7vehq": (37, 27000, 13.740523813264303, 2779.7863010689616),
}
STYLES = {
    "ixzworl4": {"color": "#2563eb", "linestyle": "-", "marker": "o"},
    "jri7vehq": {"color": "#bc6209", "linestyle": (0, (4, 3)), "marker": "s"},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_history() -> dict:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    require(data["schema_version"] == 1, "Unsupported data schema")
    require([run["run_id"] for run in data["series"]] == list(EXPECTED), "Unexpected runs/order")
    provenance = data["provenance"]
    require(provenance["metric"]["history_field"] == "val_aggregate_rmse_deg", "Wrong metric")
    require(provenance["time"]["history_field"] == "wall_clock_seconds", "Wrong clock")
    require(provenance["time"]["unit"] == "seconds", "Wrong time unit")
    for run in data["series"]:
        count, best_step, best_rmse, best_seconds = EXPECTED[run["run_id"]]
        points = run["points"]
        require(len(points) == count == run["expected_point_count"], "Wrong point count")
        steps = list(range(1000, (count + 1) * 1000, 1000))
        require([p["step"] for p in points] == steps, "Missing/reordered steps")
        for point in points:
            require(set(point) == {"step", "full30_mixed_rmse", "wall_clock_seconds"}, "Wrong fields")
            require(type(point["step"]) is int, "Step must be an integer")
            for value in point.values():
                require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                        "Non-finite/nonpositive value")
            require(point["wall_clock_seconds"] < 10000, "Time is not elapsed head-loop seconds")
        require(all(a["wall_clock_seconds"] < b["wall_clock_seconds"]
                    for a, b in zip(points, points[1:])), "Nonmonotonic times")
        best = min(points, key=lambda p: p["full30_mixed_rmse"])
        require(best == run["best"] == {
            "step": best_step, "full30_mixed_rmse": best_rmse, "wall_clock_seconds": best_seconds,
        }, "Best point changed")
        canonical = json.dumps(points, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        require(hashlib.sha256(canonical).hexdigest() == run["points_sha256"], "Numerical checksum mismatch")
        source = run["source_archive"]
        require(not Path(source["path"]).is_absolute() and ".." not in Path(source["path"]).parts,
                "Archive path must be research-repo-relative")
        require(len(source["sha256"]) == 64 and all(c in "0123456789abcdef" for c in source["sha256"]),
                "Invalid archive checksum")
    require(sum(len(run["points"]) for run in data["series"]) == 85, "Expected 85 measured points")
    return data


def check_text_bounds(fig: Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        require(box.x0 >= canvas.x0 + 2 and box.x1 <= canvas.x1 - 2
                and box.y0 >= canvas.y0 + 2 and box.y1 <= canvas.y1 - 2,
                f"Text extends outside figure: {artist.get_text()!r}")
    for ax in fig.axes:
        boxes = [text.get_window_extent(renderer) for text in ax.texts]
        require(not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]),
                "Best-point annotations overlap")


def render(data: dict) -> bytes:
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "text.color": "#172033",
        "axes.labelcolor": "#172033",
        "xtick.color": "#405068",
        "ytick.color": "#405068",
        "svg.fonttype": "none",
        "svg.hashsalt": "cosmos-t16-t2-training-v1",
    }):
        fig = Figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=72, facecolor="#fbfcfe")
        FigureCanvasAgg(fig)
        fig.text(0.5, 0.97, "Action-head learning with cached Cosmos features",
                 ha="center", va="top", fontsize=24, weight="bold")
        fig.text(0.5, 0.925, "Same Video-LoRA backbone · separate SmolExpert heads · 88 fixed validation anchors",
                 ha="center", va="top", fontsize=16, color="#405068")
        axes = [fig.add_axes((0.14, 0.545, 0.82, 0.30)),
                fig.add_axes((0.14, 0.145, 0.82, 0.30))]
        all_points = [p for run in data["series"] for p in run["points"]]
        scores = [p["full30_mixed_rmse"] for p in all_points]
        pad = max((max(scores) - min(scores)) * 0.06, 0.4)
        y_min = math.floor((min(scores) - pad) * 2) / 2
        y_max = math.ceil((max(scores) + pad) * 2) / 2
        legend_handles = []
        for panel, ax in enumerate(axes):
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(True, color="#dbe2ea", linewidth=0.7, linestyle=(0, (3, 4)))
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
            for edge in ("left", "bottom"):
                ax.spines[edge].set_color("#a5b1c0")
                ax.spines[edge].set_linewidth(0.8)
            ax.tick_params(labelsize=16, length=0, pad=9)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(list(range(math.ceil(y_min / 2) * 2, math.floor(y_max / 2) * 2 + 1, 2)))
            ax.set_ylabel("Full-30 action RMSE\n(mixed units)", fontsize=18, labelpad=17)
            field, divisor = ("step", 1000) if panel == 0 else ("wall_clock_seconds", 60)
            tick_interval = 10 if panel == 0 else 30
            x_max = math.ceil(max(p[field] / divisor for p in all_points) / tick_interval) * tick_interval
            ax.set_xlim(0, x_max)
            ax.set_xticks(list(range(0, int(x_max) + 1, tick_interval)))
            ax.set_xlabel("Optimizer steps (thousands)" if panel == 0 else "Elapsed head-training time (minutes)",
                          fontsize=18, labelpad=14)
            ax.set_title("A   Optimization steps" if panel == 0 else "B   Elapsed head-training time",
                         loc="left", fontsize=20, weight="bold", pad=19)
            for run in data["series"]:
                points = run["points"]
                style = STYLES[run["run_id"]]
                xs = [p[field] / divisor for p in points]
                ys = [p["full30_mixed_rmse"] for p in points]
                require(all(0 < x < x_max for x in xs), "Measured time/step clipped")
                require(all(y_min < y < y_max for y in ys), "Measured RMSE clipped")
                line, = ax.plot(xs, ys, label=run["label"], linewidth=1.5, markersize=3.4,
                                markeredgewidth=0.5, zorder=3, **style)
                line.set_gid(f"{run['run_id']}-{'steps' if panel == 0 else 'minutes'}")
                if panel == 0:
                    legend_handles.append(line)
                best = run["best"]
                x, y = best[field] / divisor, best["full30_mixed_rmse"]
                ax.scatter([x], [y], s=58, facecolor="white", edgecolor=style["color"], linewidth=1.5, zorder=4)
                right = run["run_id"] == "ixzworl4" if panel == 0 else run["run_id"] == "jri7vehq"
                label = (f"Best {y:.2f}\n{best['step'] / 1000:g}k steps" if panel == 0
                         else f"Best {y:.2f}\n{best['wall_clock_seconds'] / 60:.1f} min")
                ax.annotate(label, xy=(x, y), xytext=(14 if right else -14, 37),
                            textcoords="offset points", ha="left" if right else "right", va="bottom",
                            fontsize=15, color=style["color"], linespacing=1.3,
                            arrowprops={"arrowstyle": "-", "color": style["color"], "lw": 0.8}, zorder=5)
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.53, 0.902),
                   ncol=2, frameon=False, fontsize=17, handlelength=2.8, columnspacing=2.0)
        fig.text(0.52, 0.056, "Single-seed historical runs; head loop only.\nExcludes video-LoRA training and cache creation.",
                 ha="center", va="top", fontsize=16, color="#405068", linespacing=1.45)
        check_text_bounds(fig)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", facecolor=fig.get_facecolor(),
                    metadata={"Date": None, "Creator": "video-vam-figures/plot_training.py",
                              "Title": "Action-head learning with cached Cosmos features",
                              "Description": "85 measured validation points against optimizer steps and elapsed head-loop minutes. Single-seed historical runs; excludes Video-LoRA training and cache creation."})
        payload = buffer.getvalue().decode("utf-8")
        # Match intrinsic CSS pixels to the viewBox, not Matplotlib's point units.
        payload = payload.replace(f'width="{WIDTH}pt" height="{HEIGHT}pt"',
                                  f'width="{WIDTH}px" height="{HEIGHT}px"', 1)
        return ("\n".join(line.rstrip() for line in payload.splitlines()) + "\n").encode("utf-8")


def check_svg(svg: bytes, data: dict) -> None:
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    require(root.tag == ns + "svg", "Not SVG XML")
    require(root.get("viewBox") == f"0 0 {WIDTH} {HEIGHT}", "Unexpected SVG dimensions")
    require(root.get("width") == f"{WIDTH}px" and root.get("height") == f"{HEIGHT}px",
            "Unexpected intrinsic SVG pixel dimensions")
    texts = " ".join("".join(node.itertext()) for node in root.iter(ns + "text"))
    for label in ("Full-30 action RMSE", "(mixed units)", "Optimizer steps (thousands)",
                  "Elapsed head-training time (minutes)", "Best 13.06", "Best 13.74",
                  "T=16 · spatial pool2", "T=2 · unpooled (undistilled)"):
        require(label in texts, f"Missing SVG text: {label}")
    require(not any(node.tag == "{http://purl.org/dc/elements/1.1/}date" for node in root.iter()),
            "Nondeterministic SVG date")
    for run in data["series"]:
        for panel in ("steps", "minutes"):
            group = next((node for node in root.iter(ns + "g")
                          if node.get("id") == f"{run['run_id']}-{panel}"), None)
            require(group is not None, "Missing plotted series")
            require(sum(1 for _ in group.iter(ns + "use")) == len(run["points"]),
                    "SVG does not contain every measured marker")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate data, layout and exact SVG regeneration without writing")
    args = parser.parse_args()
    data = load_history()
    svg = render(data)
    check_svg(svg, data)
    if args.check:
        require(OUTPUT.read_bytes() == svg, "SVG differs; regenerate with the same Matplotlib version")
    else:
        require(OUTPUT.parent.is_dir(), "Expected site asset directory is missing")
        OUTPUT.write_bytes(svg)
    print(f"Validated 85 measured points, all SVG markers and text bounds; {WIDTH}×{HEIGHT}; Matplotlib {matplotlib.__version__}")
    print(f"{'Checked' if args.check else 'Wrote'} {OUTPUT}")
    for run in data["series"]:
        best = run["best"]
        print(f"{run['run_id']}: best {best['full30_mixed_rmse']:.14f} at {best['step']} steps; "
              f"{best['wall_clock_seconds']:.12f} s / {best['wall_clock_seconds'] / 60:.9f} min")


if __name__ == "__main__":
    main()
