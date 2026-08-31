"""Render the final CREATION ENGINE / Birds short.

The first half uses a Boids-inspired simulation. The final flock adds a
cinematic synchronized ribbon so the visual payoff reads clearly in a Short.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, sosfilt


# Video settings
W, H = 720, 1280
FPS = 30
DURATION = 41.0
BIRD_COUNT = 650
NEIGHBOR_SAMPLE = 16
SEED = 7

# Palette
GOLD = (255, 202, 88)
CYAN = (0, 234, 255)
BLUE = (53, 116, 255)
VIOLET = (155, 99, 255)
ORANGE = (255, 111, 61)
GREEN = (53, 240, 160)

SCENES = [
    (0.0, "boot"),
    (2.6, "create"),
    (6.2, "chaos"),
    (9.0, "sep"),
    (14.0, "ali"),
    (19.0, "coh"),
    (24.0, "flow"),
    (32.0, "science"),
    (37.0, "next"),
]

ACCENT = {
    "boot": GOLD,
    "create": CYAN,
    "chaos": ORANGE,
    "sep": BLUE,
    "ali": VIOLET,
    "coh": ORANGE,
    "flow": GREEN,
    "science": CYAN,
    "next": VIOLET,
}


def scene_at(t: float) -> str:
    current = "boot"
    for start, name in SCENES:
        if t >= start:
            current = name
    return current


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def find_font(kind: str) -> str | None:
    """Use common system fonts; no font files are bundled."""
    candidates = {
        "sans": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ],
        "bold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        "mono": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ],
        "monob": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "C:/Windows/Fonts/consolab.ttf",
            "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        ],
    }
    for candidate in candidates[kind]:
        if Path(candidate).exists():
            return candidate
    return None


def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def simulate_boids() -> tuple[np.ndarray, np.ndarray]:
    """Simulate the build-up: chaos -> separation -> alignment -> cohesion."""
    rng = np.random.default_rng(SEED)
    frames = int(FPS * DURATION)
    dt = 1 / FPS

    center = np.array([W / 2, H * 0.39], np.float32)
    angle = rng.uniform(0, 2 * np.pi, BIRD_COUNT)
    radius = rng.uniform(0, 16, BIRD_COUNT)
    pos = np.c_[
        center[0] + np.cos(angle) * radius,
        center[1] + np.sin(angle) * radius,
    ].astype(np.float32)

    vel_angle = rng.uniform(0, 2 * np.pi, BIRD_COUNT)
    speed = rng.uniform(45, 95, BIRD_COUNT)
    vel = np.c_[np.cos(vel_angle) * speed, np.sin(vel_angle) * speed].astype(np.float32)

    # Each bird samples a small fixed set of neighbors for speed.
    neighbors = np.empty((BIRD_COUNT, NEIGHBOR_SAMPLE), np.int32)
    for i in range(BIRD_COUNT):
        q = rng.choice(BIRD_COUNT - 1, NEIGHBOR_SAMPLE, False)
        neighbors[i] = np.where(q >= i, q + 1, q)

    positions = np.zeros((frames, BIRD_COUNT, 2), np.float32)
    active_counts = np.zeros(frames, np.int32)

    for frame in range(frames):
        t = frame / FPS
        scene = scene_at(t)

        if t < 2.6:
            active = 0
        elif t < 5.1:
            progress = max(0.0, min(1.0, (t - 2.6) / 2.5))
            active = int(BIRD_COUNT * (1 - (1 - progress) ** 2))
        else:
            active = BIRD_COUNT
        active_counts[frame] = active

        if active:
            p = pos[:active]
            v = vel[:active]
            acc = np.zeros_like(p)

            # Keep the flock inside the cinematic safe region.
            acc[:, 0] += np.where(p[:, 0] < 75, 75 - p[:, 0], 0)
            acc[:, 0] += np.where(p[:, 0] > W - 75, -(p[:, 0] - (W - 75)), 0)
            acc[:, 1] += np.where(p[:, 1] < H * 0.22, H * 0.22 - p[:, 1], 0)
            acc[:, 1] += np.where(p[:, 1] > H * 0.78, -(p[:, 1] - H * 0.78), 0)

            if scene in ("create", "chaos", "sep", "ali"):
                acc += (np.array([W / 2, H * 0.48]) - p) * 0.035

            idx = neighbors[:active]
            valid = idx < active
            clipped = idx.clip(max=max(active - 1, 0))
            neighbor_pos = pos[clipped]
            neighbor_vel = vel[clipped]
            delta = p[:, None, :] - neighbor_pos
            dist2 = (delta * delta).sum(2) + 1e-4

            # Separation: avoid crowding nearby birds.
            if t >= 9:
                mask = valid & (dist2 < 26**2)
                repulsion = np.where(mask[:, :, None], delta / dist2[:, :, None], 0).sum(1)
                desired = normalize(repulsion) * 118 - v
                mag = np.linalg.norm(desired, axis=1, keepdims=True)
                desired *= np.minimum(1, 92 / np.maximum(mag, 1e-6))
                acc += desired * 1.48

            # Alignment: match nearby flight direction.
            if t >= 14:
                mask = valid & (dist2 < 68**2)
                count = np.maximum(mask.sum(1, keepdims=True), 1)
                avg_vel = (neighbor_vel * mask[:, :, None]).sum(1) / count
                desired = normalize(avg_vel) * 118 - v
                mag = np.linalg.norm(desired, axis=1, keepdims=True)
                desired *= np.minimum(1, 75 / np.maximum(mag, 1e-6))
                acc += desired * 1.18

            # Cohesion: move toward nearby flock mates.
            if t >= 19:
                mask = valid & (dist2 < 82**2)
                count = np.maximum(mask.sum(1, keepdims=True), 1)
                avg_pos = (neighbor_pos * mask[:, :, None]).sum(1) / count
                desired = normalize(avg_pos - p) * 112 - v
                mag = np.linalg.norm(desired, axis=1, keepdims=True)
                desired *= np.minimum(1, 58 / np.maximum(mag, 1e-6))
                acc += desired * 0.95

                # Extra cinematic cohesion after rule 3.
                group_center = p.mean(axis=0)
                acc += (group_center - p) * 0.16
                acc += (np.array([W / 2, H * 0.50]) - group_center) * 0.055

            # A gentle global flow helps bridge into the synchronized finale.
            if t >= 24:
                flow = np.array(
                    [
                        W / 2 + math.sin(t * 0.62) * W * 0.14 + math.sin(t * 1.31 + 1) * W * 0.035,
                        H * 0.49 + math.cos(t * 0.72) * H * 0.065 + math.cos(t * 1.8 + 0.5) * H * 0.02,
                    ]
                )
                radial = flow - p
                desired = normalize(radial) * 105 - v
                mag = np.linalg.norm(desired, axis=1, keepdims=True)
                desired *= np.minimum(1, 48 / np.maximum(mag, 1e-6))
                acc += desired * 0.60
                acc += radial * 0.095
                tangent = np.c_[-radial[:, 1], radial[:, 0]]
                acc += normalize(tangent) * 13

            jitter = 36 if scene == "chaos" else (18 if scene in ("create", "sep") else 8)
            acc += rng.normal(0, jitter, (active, 2)).astype(np.float32)

            vel[:active] += acc * dt
            if t >= 19:
                vel[:active] *= 0.993
            velocity_mag = np.linalg.norm(vel[:active], axis=1, keepdims=True)
            vel[:active] *= np.minimum(1, 118 / np.maximum(velocity_mag, 1e-6))
            pos[:active] += vel[:active] * dt
            pos[:active, 0] = np.clip(pos[:active, 0], 3, W - 3)
            pos[:active, 1] = np.clip(pos[:active, 1], H * 0.20, H * 0.80)

        positions[frame] = pos

    return positions, active_counts


def cinematic_identity() -> tuple[np.ndarray, np.ndarray]:
    """Stable per-bird coordinates used only for the synchronized finale."""
    rng = np.random.default_rng(2026)
    u = np.linspace(-1, 1, BIRD_COUNT, dtype=np.float32)
    u += rng.normal(0, 0.012, BIRD_COUNT).astype(np.float32)
    v = np.clip(rng.normal(0, 0.78, BIRD_COUNT), -1.8, 1.8).astype(np.float32)
    return u, v


U, V = cinematic_identity()


def cinematic_shape(t: float, active: int) -> np.ndarray:
    """Create one deforming ribbon that banks and travels as a single flock."""
    u = U[:active]
    v = V[:active]
    tau = max(0.0, t - 24.0)

    breathe = 1.0 + 0.055 * math.sin(tau * 1.35)
    base_x = (182 * u) * breathe
    base_y = 48 * np.sin(np.pi * 1.15 * u + tau * 0.34)
    base_y += 14 * np.sin(np.pi * 2.7 * u - tau * 0.24)

    dydx = 48 * np.pi * 1.15 * np.cos(np.pi * 1.15 * u + tau * 0.34)
    dydx += 14 * np.pi * 2.7 * np.cos(np.pi * 2.7 * u - tau * 0.24)
    dx = np.full_like(dydx, 182.0 * breathe)
    normal_mag = np.sqrt(dx * dx + dydx * dydx)
    nx, ny = -dydx / normal_mag, dx / normal_mag

    thickness = (15 + 30 * (1 - u * u)) * (1.0 + 0.06 * math.sin(tau * 1.1))
    offset = v * thickness
    x = base_x + nx * offset
    y = base_y + ny * offset

    # Shared path and turn make the final movement visibly synchronized.
    center_x = W * 0.50 + 112 * math.sin(tau * 0.48)
    center_y = H * 0.49 + 46 * math.sin(tau * 0.66 + 0.45)
    theta = 0.34 * math.sin(tau * 0.52) + 0.10 * math.sin(tau * 1.05 + 0.6)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot_x = x * cos_t - y * sin_t
    rot_y = x * sin_t + y * cos_t
    rot_y *= 1.0 - 0.10 * abs(math.sin(theta))

    return np.c_[center_x + rot_x, center_y + rot_y].astype(np.float32)


def display_positions(positions: np.ndarray, frame: int, active: int) -> np.ndarray:
    t = frame / FPS
    p = positions[frame, :active].copy()
    if t >= 24 and active:
        blend = np.clip((t - 24) / 2.7, 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        p = p * (1 - blend) + cinematic_shape(t, active) * blend
    return p


def build_overlays() -> dict[str, np.ndarray]:
    sans = find_font("sans")
    bold = find_font("bold")
    mono = find_font("mono")
    monob = find_font("monob")

    def fit(draw, xy, text, width, font_path, size, fill):
        while size > 14:
            font = load_font(font_path, size)
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= width:
                break
            size -= 1
        draw.text(xy, text, font=font, fill=fill)

    code_map = {
        "boot": "SYSTEM BOOT...\n> initializing life",
        "create": '> create("birds");\n> count = 650;',
        "chaos": "> run();\nERROR: chaos",
        "sep": "+ separation();\n// keep personal space",
        "ali": "+ alignment();\n// match nearby direction",
        "coh": "+ cohesion();\n// stay with the flock",
        "flow": "> render_flock();\nBUILD SUCCESSFUL ✓",
        "science": "computer_science:\n> BOIDS ALGORITHM",
        "next": 'CREATION COMPLETE ✓\n> next("ants");',
    }

    lower_map = {
        "create": ("CREATION", "A flock begins.", "Not one bird at a time — one system.", CYAN),
        "chaos": ("PROBLEM", "Too much chaos.", "The birds exist, but nothing coordinates them.", ORANGE),
        "sep": ("RULE 01", "Don’t get too close.", "Nearby birds gently push apart.", BLUE),
        "ali": ("RULE 02", "Match direction.", "Each bird adjusts toward its neighbors.", VIOLET),
        "coh": ("RULE 03", "Stay together.", "Local attraction keeps the flock connected.", ORANGE),
        "flow": ("RESULT", "No leader.", "Hundreds of local decisions become one pattern.", GREEN),
        "science": ("EMERGENCE", "Simple rules. Complex behavior.", "Separation · Alignment · Cohesion", CYAN),
        "next": ("NEXT BUILD", "ANTS", "What could simple rules create next?", VIOLET),
    }

    overlays: dict[str, np.ndarray] = {}
    for _, scene_name in SCENES:
        accent = ACCENT[scene_name]
        image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        draw.text((47, 57), "CREATION", font=load_font(monob, 15), fill=GOLD + (235,))
        draw.text((139, 57), " ENGINE", font=load_font(monob, 15), fill=(230, 238, 250, 170))
        draw.multiline_text(
            (672, 57),
            "EARTH / DAY 5\nBUILD 006",
            font=load_font(mono, 12),
            fill=(210, 222, 240, 105),
            anchor="ra",
            spacing=3,
            align="right",
        )

        x0, y0, x1, y1 = 47, 105, 673, 198
        draw.rounded_rectangle((x0, y0, x1, y1), 16, fill=(3, 6, 18, 180), outline=accent + (120,), width=2)
        for i in range(3):
            draw.ellipse((x0 + 16 + i * 13, y0 + 14, x0 + 22 + i * 13, y0 + 20), fill=(255, 255, 255, 55))

        for line_no, line in enumerate(code_map[scene_name].split("\n")):
            color = (235, 245, 255, 230)
            if "ERROR" in line:
                color = ORANGE + (245,)
            elif "SUCCESS" in line:
                color = GREEN + (245,)
            elif line.startswith("//") or "SYSTEM" in line or "computer_science" in line:
                color = (150, 169, 198, 195)
            font_path = mono if line.startswith("//") else monob
            draw.text((x0 + 17, y0 + 34 + line_no * 25), line, font=load_font(font_path, 16), fill=color)

        if scene_name in lower_map:
            kicker, title, subtitle, lower_accent = lower_map[scene_name]
            lx0, ly0, lx1, ly1 = 47, 1010, 673, 1213
            draw.rounded_rectangle((lx0, ly0, lx1, ly1), 22, fill=(3, 6, 18, 228), outline=(255, 255, 255, 30), width=1)
            draw.rounded_rectangle((lx0, ly0, lx0 + 6, ly1), 3, fill=lower_accent + (255,))
            draw.text((lx0 + 25, ly0 + 22), kicker, font=load_font(monob, 15), fill=lower_accent + (255,))
            fit(draw, (lx0 + 25, ly0 + 66), title, lx1 - lx0 - 52, bold, 42, (250, 252, 255, 255))
            fit(draw, (lx0 + 25, ly0 + 128), subtitle, lx1 - lx0 - 52, sans, 22, (215, 226, 242, 210))

        overlays[scene_name] = np.array(image)

    return overlays


def build_backgrounds() -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:H, 0:W]
    stars = [(int((i * 89.7) % W), int(((i * 153.3) % (H * 0.73)) + H * 0.08), 1 if i % 7 else 2) for i in range(58)]
    backgrounds = {}

    for scene_name, color in ACCENT.items():
        accent_bgr = np.array(color[::-1], np.float32)
        grad_y = (yy / (H - 1))[..., None]
        top = np.array([21, 8, 3], np.float32)
        bottom = np.array([10, 3, 2], np.float32)
        bg = top * (1 - grad_y) + bottom * grad_y

        distance = np.sqrt(((xx - W * 0.5) / (W * 0.70)) ** 2 + ((yy - H * 0.52) / (H * 0.48)) ** 2)
        wash = np.clip(1 - distance, 0, 1)[..., None] * 0.14
        bg = bg * (1 - wash) + accent_bgr * wash
        frame = np.clip(bg, 0, 255).astype(np.uint8)

        for x, y, radius in stars:
            cv2.circle(frame, (x, y), radius, (205, 215, 230), -1, cv2.LINE_AA)

        # Abstract golden presence: intentionally not a literal figure of God.
        divine_dist = np.sqrt((xx - W * 0.5) ** 2 + (yy - H * 0.19) ** 2)
        glow_strength = 0.48 if scene_name not in ("flow", "science") else 0.28
        glow = np.exp(-(divine_dist / 55) ** 2)[..., None] * glow_strength
        gold_bgr = np.array([90, 190, 255], np.float32)
        frame = (frame.astype(np.float32) * (1 - glow) + gold_bgr * glow).clip(0, 255).astype(np.uint8)
        cv2.circle(frame, (W // 2, int(H * 0.19)), 10, (205, 246, 255), -1, cv2.LINE_AA)

        # Narrow beam connects the presence to the creation area.
        x_dist = np.abs(xx - W * 0.5)
        y_mask = (yy >= H * 0.20) & (yy <= H * 0.39)
        beam = (np.exp(-(x_dist / 5) ** 2) * y_mask * 0.12)[..., None]
        frame = (frame.astype(np.float32) * (1 - beam) + np.array([120, 205, 255], np.float32) * beam).clip(0, 255).astype(np.uint8)

        if scene_name == "create":
            creation_dist = np.sqrt((xx - W * 0.5) ** 2 + (yy - H * 0.39) ** 2)
            creation_glow = np.exp(-(creation_dist / 80) ** 2)[..., None] * 0.22
            frame = (
                frame.astype(np.float32) * (1 - creation_glow)
                + np.array([255, 210, 60], np.float32) * creation_glow
            ).clip(0, 255).astype(np.uint8)

        backgrounds[scene_name] = frame

    return backgrounds


def render_video(output_path: Path, positions: np.ndarray, active_counts: np.ndarray) -> None:
    overlays = build_overlays()
    backgrounds = build_backgrounds()
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames = int(FPS * DURATION)

    for frame_no in range(frames):
        t = frame_no / FPS
        scene_name = scene_at(t)
        frame = backgrounds[scene_name].copy()

        # Orbiting motes reinforce the abstract "creative presence".
        for i in range(11):
            angle = t * 0.22 + i * 2 * np.pi / 11
            radius = 24 + (i % 3) * 7
            x = int(W * 0.5 + math.cos(angle) * radius)
            y = int(H * 0.19 + math.sin(angle * 1.12) * radius * 0.55)
            cv2.circle(frame, (x, y), 1 + (i % 2), (120, 215, 255), -1, cv2.LINE_AA)

        active = int(active_counts[frame_no])
        if active:
            p = display_positions(positions, frame_no, active)
            p_next = display_positions(positions, min(frame_no + 1, frames - 1), active)
            velocity = p_next - p
            mag = np.sqrt((velocity * velocity).sum(1))
            mag = np.where(mag < 1e-6, 1, mag)
            ux, uy = velocity[:, 0] / mag, velocity[:, 1] / mag
            px, py = -uy, ux
            size = 4.0

            tip = np.c_[p[:, 0] + ux * size * 0.8, p[:, 1] + uy * size * 0.8]
            left = np.c_[p[:, 0] - ux * size * 0.7 + px * size * 1.4, p[:, 1] - uy * size * 0.7 + py * size * 1.4]
            right = np.c_[p[:, 0] - ux * size * 0.7 - px * size * 1.4, p[:, 1] - uy * size * 0.7 - py * size * 1.4]
            bird_lines = np.stack([left, tip, right], 1).astype(np.int32)

            bird_color = {
                "chaos": (214, 226, 255),
                "sep": (255, 225, 205),
                "ali": (255, 219, 232),
                "coh": (207, 222, 255),
                "flow": (248, 255, 230),
                "science": (248, 255, 230),
            }.get(scene_name, (255, 245, 238))
            cv2.polylines(frame, list(bird_lines), False, bird_color, 1, cv2.LINE_AA)

        # Small white flash on scene changes.
        for scene_start, _ in SCENES[1:]:
            delta = t - scene_start
            if 0 <= delta < 0.16:
                alpha = (1 - delta / 0.16) * 0.18
                frame = cv2.addWeighted(frame, 1 - alpha, np.full_like(frame, 255), alpha, 0)
                break

        overlay = overlays[scene_name]
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255
        rgb = overlay[:, :, :3][:, :, ::-1].astype(np.float32)
        frame = (frame * (1 - alpha) + rgb * alpha).astype(np.uint8)
        writer.write(frame)

        if frame_no % 180 == 0:
            print(f"render {frame_no}/{frames}")

    writer.release()


def synthesize_soundtrack(output_path: Path) -> None:
    """Generate the original procedural soundtrack; no external audio files."""
    rng = np.random.default_rng(SEED)
    sample_rate = 48_000
    total = int(DURATION * sample_rate)
    left = np.zeros(total)
    right = np.zeros(total)

    def tone(start, duration, freq, amp=0.2, kind="sine", pan=0, attack=0.02, release=0.15, glide=None):
        i0 = max(0, int(start * sample_rate))
        i1 = min(total, int((start + duration) * sample_rate))
        count = i1 - i0
        if count < 2:
            return
        x = np.arange(count) / sample_rate
        if glide:
            f0, f1 = glide
            phase = 2 * np.pi * (f0 * x + (f1 - f0) / (2 * duration) * x * x)
        else:
            phase = 2 * np.pi * freq * x
        if kind == "triangle":
            signal = 2 / np.pi * np.arcsin(np.sin(phase))
        elif kind == "saw":
            signal = 2 * ((phase / (2 * np.pi)) % 1) - 1
        else:
            signal = np.sin(phase)

        env = np.ones(count)
        a = min(count, int(attack * sample_rate))
        r = min(count, int(release * sample_rate))
        if a:
            env[:a] = np.linspace(0, 1, a)
        if r:
            env[-r:] *= np.linspace(1, 0, r)
        signal *= env * amp
        lg = math.sqrt((1 - pan) / 2)
        rg = math.sqrt((1 + pan) / 2)
        left[i0:i1] += signal * lg
        right[i0:i1] += signal * rg

    def noise(start, duration, amp=0.15, low=200, high=5000, pan=0):
        i0 = max(0, int(start * sample_rate))
        i1 = min(total, int((start + duration) * sample_rate))
        count = i1 - i0
        if count < 2:
            return
        signal = rng.normal(0, 1, count)
        nyquist = sample_rate / 2
        sos = butter(3, [max(1, low) / nyquist, min(high, nyquist - 100) / nyquist], btype="band", output="sos")
        signal = sosfilt(sos, signal)
        signal = signal / (np.max(np.abs(signal)) + 1e-9)
        signal *= amp * (np.sin(np.linspace(0, np.pi, count)) ** 1.35)
        lg = math.sqrt((1 - pan) / 2)
        rg = math.sqrt((1 + pan) / 2)
        left[i0:i1] += signal * lg
        right[i0:i1] += signal * rg

    def impact(t0, root=52, amp=0.34):
        tone(t0, 0.75, root, amp, "sine", 0, 0.002, 0.6)
        tone(t0, 0.34, root * 2, 0.14, "triangle", -0.08, 0.002, 0.3)
        noise(t0, 0.22, 0.20, 70, 1400, 0.05)

    def sparkle(t0, base=660, amp=0.11):
        for k, (multiplier, pan) in enumerate([(1, -0.35), (1.25, 0.2), (1.5, 0.45)]):
            tone(t0 + k * 0.07, 0.22, base * multiplier, amp * (1 - 0.16 * k), "sine", pan, 0.004, 0.18)

    # Cosmic bed.
    for freq, amp, pan in [(55, 0.075, -0.15), (82.41, 0.05, 0.15), (110, 0.026, 0)]:
        tone(0, DURATION, freq, amp, "sine", pan, 1, 1.5)
    for t0 in np.arange(0, DURATION, 4):
        tone(t0, 2.8, 164.81, 0.025, "sine", -0.2, 0.45, 1.2)
        tone(t0 + 0.6, 2.2, 220, 0.02, "sine", 0.22, 0.5, 1)

    # Scene cues.
    impact(0.1, 42, 0.3)
    sparkle(0.25, 420, 0.07)
    noise(2.55, 0.9, 0.22, 300, 7000, -0.1)
    tone(2.55, 0.85, 120, 0.17, "saw", 0, 0.01, 0.55, (120, 1500))
    sparkle(2.72, 520, 0.13)
    impact(6.15, 48, 0.31)
    noise(6.2, 1.8, 0.12, 100, 3000, 0.15)

    for t0, freq, pan in [(6.35, 86, -0.3), (6.95, 121, 0.25), (7.55, 69, -0.1), (8.2, 104, 0.35)]:
        tone(t0, 0.3, freq, 0.12, "saw", pan, 0.01, 0.22)

    for t0, root, sparkle_freq in [(9, 58, 440), (14, 65.4, 523.25), (19, 73.42, 659.25)]:
        impact(t0, root, 0.38)
        sparkle(t0 + 0.08, sparkle_freq, 0.14)
        noise(t0, 0.38, 0.10, 800, 8500, 0.1)

    tone(23.7, 1.2, 110, 0.16, "saw", 0, 0.02, 0.9, (110, 1700))
    impact(24, 49, 0.42)
    for freq, amp, pan in [(220, 0.09, -0.35), (277.18, 0.08, -0.1), (329.63, 0.08, 0.15), (440, 0.06, 0.35)]:
        tone(24, 6.5, freq, amp, "sine", pan, 0.6, 2.2)
    noise(24, 1.25, 0.18, 500, 9000, 0.05)
    for t0 in np.arange(25, 31.5, 1.1):
        sparkle(float(t0), 780 + 40 * math.sin(t0), 0.035)

    impact(32, 55, 0.3)
    sparkle(32.1, 784, 0.14)
    for freq, amp, pan in [(261.63, 0.055, -0.25), (329.63, 0.05, 0), (392, 0.045, 0.25)]:
        tone(32, 4.7, freq, amp, "sine", pan, 0.35, 1.6)

    impact(37, 46, 0.25)
    tone(37.15, 0.24, 110, 0.15, "sine", -0.2, 0.005, 0.18)
    tone(37.34, 0.28, 164.81, 0.14, "sine", 0, 0.005, 0.2)
    tone(37.58, 0.75, 220, 0.17, "triangle", 0.25, 0.008, 0.6)
    noise(37, 0.5, 0.08, 900, 8000, 0.25)
    tone(38, 3, 110, 0.035, "sine", 0, 0.3, 1.7)

    stereo = np.tanh(np.stack([left, right], 1) * 1.45)
    stereo = stereo / (np.max(np.abs(stereo)) + 1e-9) * 0.93
    sf.write(output_path, stereo, sample_rate, subtype="PCM_16")


def mux(video: Path, audio: Path, output: Path, audio_db: float = 0.0) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    volume = 10 ** (audio_db / 20)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-filter:a",
        f"volume={volume:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CREATION ENGINE — Birds")
    parser.add_argument("--output-dir", default="output", help="Folder for rendered files")
    parser.add_argument("--skip-audio", action="store_true", help="Render only the silent MP4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    silent = output_dir / "creation_engine_birds_silent.mp4"
    soundtrack = output_dir / "creation_engine_birds_soundtrack.wav"
    final = output_dir / "creation_engine_birds_final.mp4"
    voice_ready = output_dir / "creation_engine_birds_voice_ready.mp4"

    print("Simulating birds...")
    positions, active_counts = simulate_boids()
    print("Rendering video...")
    render_video(silent, positions, active_counts)

    if not args.skip_audio:
        print("Synthesizing soundtrack...")
        synthesize_soundtrack(soundtrack)
        print("Muxing final versions...")
        mux(silent, soundtrack, final, audio_db=0.0)
        mux(silent, soundtrack, voice_ready, audio_db=-6.0)

    print(f"Done. Files are in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
