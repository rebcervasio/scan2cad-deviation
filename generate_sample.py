"""Generate sample mechanical part STLs for the demo (no boolean ops required)."""
import numpy as np
import trimesh
import os


def make_mechanical_part():
    """
    Build a mechanical part by concatenating primitives — no boolean ops needed.
    Produces a recognizable part: base plate + vertical rib + cylinder boss.
    """
    parts = []

    # Base plate
    plate = trimesh.creation.box(extents=[80, 50, 8])
    plate.apply_translation([40, 25, 4])
    parts.append(plate)

    # Vertical rib/wall
    rib = trimesh.creation.box(extents=[8, 50, 40])
    rib.apply_translation([4, 25, 28])
    parts.append(rib)

    # Cylindrical boss on the plate (mounting feature)
    boss = trimesh.creation.cylinder(radius=10, height=12, sections=32)
    boss.apply_translation([55, 25, 14])
    parts.append(boss)

    # Second smaller boss
    boss2 = trimesh.creation.cylinder(radius=6, height=10, sections=32)
    boss2.apply_translation([35, 12, 13])
    parts.append(boss2)

    # Stiffening gusset
    gusset = trimesh.creation.box(extents=[15, 50, 15])
    gusset.apply_translation([11.5, 25, 15.5])
    parts.append(gusset)

    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    part = make_mechanical_part()
    path = "assets/sample_bracket.stl"
    part.export(path)
    print(f"Saved {path} — {len(part.vertices)} vertices, {len(part.faces)} faces")
    print(f"Bounding box: {part.bounds}")
