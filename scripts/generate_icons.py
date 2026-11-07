import math
from pathlib import Path
from PIL import Image, ImageDraw

def render_synapse_icon(size: int) -> Image.Image:
    # 4x supersampling for ultra-crisp antialiasing
    scale = 4
    canvas_size = size * scale
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Center and scaling
    cx, cy = canvas_size / 2, canvas_size / 2
    r_box = canvas_size * 0.46
    corner_radius = canvas_size * 0.22

    # Squircle background
    box = [cx - r_box, cy - r_box, cx + r_box, cy + r_box]
    draw.rounded_rectangle(box, radius=corner_radius, fill=(11, 15, 25, 255), outline=(34, 211, 238, 200), width=max(1, int(3 * scale)))

    # Inner hexagon nodes
    center_hex_r = canvas_size * 0.12
    hex_points = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        px = cx + center_hex_r * math.cos(angle)
        py = cy + center_hex_r * math.sin(angle)
        hex_points.append((px, py))

    # Outer synaptic nodes
    outer_nodes = []
    outer_r = canvas_size * 0.32
    for i in range(6):
        angle = math.radians(60 * i)
        px = cx + outer_r * math.cos(angle)
        py = cy + outer_r * math.sin(angle)
        outer_nodes.append((px, py))

    # Draw connectors
    line_w = max(2, int(2.5 * scale))
    for i in range(6):
        draw.line([hex_points[i], outer_nodes[i]], fill=(6, 182, 212, 240), width=line_w)
        # Peripheral branches
        next_i = (i + 1) % 6
        draw.line([hex_points[i], hex_points[next_i]], fill=(99, 102, 241, 220), width=line_w)

    # Central glowing node
    draw.polygon(hex_points, fill=(15, 23, 42, 255), outline=(56, 189, 248, 255))
    center_dot_r = canvas_size * 0.04
    draw.ellipse([cx - center_dot_r, cy - center_dot_r, cx + center_dot_r, cy + center_dot_r], fill=(56, 189, 248, 255))

    # Outer memory terminals
    dot_r = max(2, int(4 * scale))
    for px, py in outer_nodes:
        draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r], fill=(34, 211, 238, 255), outline=(99, 102, 241, 255), width=max(1, scale))

    # Downsample back to target size with high-quality Lanczos filter
    return img.resize((size, size), Image.Resampling.LANCZOS)

def main():
    root = Path(__file__).resolve().parent.parent
    assets_dir = root / "assets"
    ext_icons_dir = root / "extension" / "icons"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ext_icons_dir.mkdir(parents=True, exist_ok=True)

    sizes = [16, 32, 48, 64, 128, 256, 512]
    for sz in sizes:
        icon_img = render_synapse_icon(sz)
        # Save in assets
        icon_img.save(assets_dir / f"icon-{sz}.png")
        # Save in extension icons if appropriate
        if sz in [16, 32, 48, 128]:
            icon_img.save(ext_icons_dir / f"icon-{sz}.png")

    # Main icon.png
    main_512 = render_synapse_icon(512)
    main_512.save(assets_dir / "icon.png")
    print("✔ All icons rendered and exported successfully!")

if __name__ == "__main__":
    main()
