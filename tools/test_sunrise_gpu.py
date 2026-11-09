#!/usr/bin/env python3
"""Run the sunrise-sector WGSL equivalence test on a host Vulkan/DX12 GPU.

The test builds a small temporary Rust crate in .local/android/sunrise-gpu-tests.
It extracts the two sunrise functions from the production terrain.wgsl, so the
test exercises the shipped shader code rather than a duplicated implementation.
No game, launcher, prepared resources, or player profile is opened.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local" / "android" / "sunrise-gpu-tests"
TERRAIN = ROOT / "crates" / "horizon-client-render" / "src" / "terrain.wgsl"
FAN = ROOT / "crates" / "horizon-client-render" / "src" / "gfx" / "sunrise_fan.rs"
GPU_TEST = ROOT / "tools" / "android_sunrise_gpu_test.rs"


def main() -> None:
    for path in (TERRAIN, FAN, GPU_TEST):
        if not path.is_file():
            raise SystemExit(f"Required sunrise test input is missing: {path}")

    terrain = TERRAIN.read_text(encoding="utf-8")
    start = terrain.find("fn sunrise_weight_full(")
    end = terrain.find("// The sky's colour", start)
    if start < 0 or end < 0:
        raise SystemExit(
            "Could not locate the sunrise reference/optimized functions in terrain.wgsl"
        )

    work_src = WORK / "src"
    work_src.mkdir(parents=True, exist_ok=True)

    (WORK / "Cargo.toml").write_text(
        """[package]
name = "sunrise-gpu-equivalence"
version = "0.0.0"
edition = "2024"

[workspace]

[dependencies]
wgpu = { version = "30.0.1", default-features = false, features = ["std", "parking_lot", "vulkan", "dx12", "wgsl"] }
pollster = "0.4"
bytemuck = "1"
glam = "0.30"
""",
        encoding="utf-8",
    )

    (work_src / "lib.rs").write_text(
        "#![allow(dead_code)]\n"
        f'#[path = "{FAN.as_posix()}"] mod sunrise_fan;\n'
        f'#[path = "{GPU_TEST.as_posix()}"] mod sunrise_gpu_test;\n',
        encoding="utf-8",
    )

    prelude = """struct Globals {
    sunrise: vec4<f32>,
    sunrise_fan: array<vec4<f32>, 32>,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> results: array<vec2<f32>>;
"""
    compute_entry = """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&rays) {
        let direction = rays[id.x].xyz;
        results[id.x] = vec2(
            sunrise_weight_full(direction),
            sunrise_weight(direction),
        );
    }
}
"""
    (work_src / "sunrise.wgsl").write_text(
        prelude + terrain[start:end] + compute_entry,
        encoding="utf-8",
    )

    subprocess.run(
        [
            "cargo",
            "test",
            "--manifest-path",
            str(WORK / "Cargo.toml"),
            "--",
            "--nocapture",
            "--test-threads=1",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(f"Could not run the Rust test harness: {error}") from error
