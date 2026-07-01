"""Generate sample mechanical part STLs for the demo (no boolean ops required)."""
import numpy as np
import trimesh
import os


def make_mechanical_part():
    """Bracket-like part built from boxes and cylinders (plate, rib, two bosses, a gusset)."""
    plate = trimesh.creation.box(extents=[80, 50, 8])
    plate.apply_translation([40, 25, 4])

    rib = trimesh.creation.box(extents=[8, 50, 40])
    rib.apply_translation([4, 25, 28])

    boss = trimesh.creation.cylinder(radius=10, height=12, sections=32)
    boss.apply_translation([55, 25, 14])

    boss2 = trimesh.creation.cylinder(radius=6, height=10, sections=32)
    boss2.apply_translation([35, 12, 13])

    gusset = trimesh.creation.box(extents=[15, 50, 15])
    gusset.apply_translation([11.5, 25, 15.5])

    return trimesh.util.concatenate([plate, rib, boss, boss2, gusset])


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    part = make_mechanical_part()
    path = "assets/sample_bracket.stl"
    part.export(path)
    print(f"Saved {path}, {len(part.vertices)} vertices, {len(part.faces)} faces")
    print(f"Bounding box: {part.bounds}")
