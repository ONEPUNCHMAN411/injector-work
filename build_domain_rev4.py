"""
REV4 geometry: REV3's 72-open-bore pattern, plus two design changes:
  1. Outer ring (R=60mm, 48 positions incl. 24 coax) canted outward by CANT_DEG
     to actively sweep the R60-90mm dead zone that fed the far-field oxidizer-rich
     recirculation seen in REV3.
  2. LOX annulus narrowed (CUP_D 8.0 -> 7.5mm) to raise the momentum-flux ratio J
     from REV3's 0.97 toward the coaxial-swirl target range (1-4).
  3. LOX inlet exported as 36 SEPARATE patches (one per coaxial element) so each
     can carry its own local swirl boundary condition (swirlFlowRateInletVelocity
     needs a per-element origin/axis; a single combined patch can't do that).
Own from-scratch implementation, trimesh/manifold3d, same pattern as build_domain_coax.py.
"""
import os
import numpy as np
import trimesh

CHAMBER_R = 90.0
DOMAIN_LEN = 200.0

RP1_D = 2.42
RP1_RING1_R = 35.0
RP1_RING1_N = 24
RP1_RING2_R = 60.0
RP1_RING2_N = 48
COAX_RING1_N = 12
COAX_RING2_N = 24

TUBE_OD = 6.24
CUP_D = 7.5          # narrowed from REV3's 8.0mm -> raises J
CANT_DEG = 7.0        # outer ring outward cant

STUB_UP = -6.0
STUB_DOWN = 0.05

HOME = os.environ.get('HOME', '/root')
GEOM_DIR = f'{HOME}/injector_work/geom'
os.makedirs(GEOM_DIR, exist_ok=True)


def ring_positions(n, r):
    out = []
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        out.append((r * np.cos(ang), r * np.sin(ang), ang))
    return out


rp1_ring1_all = ring_positions(RP1_RING1_N, RP1_RING1_R)
rp1_ring2_all = ring_positions(RP1_RING2_N, RP1_RING2_R)
coax_ring1 = ring_positions(COAX_RING1_N, RP1_RING1_R)
coax_ring2 = ring_positions(COAX_RING2_N, RP1_RING2_R)


def is_coax(pos, coax_list, tol=1e-3):
    return any(abs(pos[0] - c[0]) < tol and abs(pos[1] - c[1]) < tol for c in coax_list)


rp1_plain = [p for p in rp1_ring1_all if not is_coax((p[0], p[1]), [(c[0], c[1]) for c in coax_ring1])] + \
            [p for p in rp1_ring2_all if not is_coax((p[0], p[1]), [(c[0], c[1]) for c in coax_ring2])]

# build element registry
elements = []
for (x, z, ang) in coax_ring1:
    elements.append(dict(x=x, z=z, ang=ang, canted=False, has_lox=True))
for (x, z, ang) in coax_ring2:
    elements.append(dict(x=x, z=z, ang=ang, canted=True, has_lox=True))
for (x, z, ang) in rp1_plain:
    r = np.hypot(x, z)
    elements.append(dict(x=x, z=z, ang=ang, canted=(r > 45.0), has_lox=False))

n_rp1 = len(elements)
lox_elements = [e for e in elements if e['has_lox']]
n_lox = len(lox_elements)
n_canted = sum(e['canted'] for e in elements)
print(f"RP1 elements: {n_rp1} (expect 72)   LOX elements: {n_lox} (expect 36)   canted: {n_canted} (expect 48)")


def cyl_along_y(radius, y0, y1, sections=32):
    h = y1 - y0
    m = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    Rx = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    m.apply_transform(Rx)
    m.apply_translation([0, (y0 + y1) / 2.0, 0])
    return m


def annulus_along_y(r_outer, r_inner, y0, y1, sections=32):
    outer = cyl_along_y(r_outer, y0, y1, sections=sections)
    inner = cyl_along_y(r_inner + 0.001, y0 - 1, y1 + 1, sections=sections)
    return trimesh.boolean.difference([outer, inner], engine='manifold')


def cant_transform(ang, canted):
    """Rotation matrix tilting local +Y outward by CANT_DEG around the local
    tangential axis, for an element at azimuth `ang`. Identity if not canted."""
    if not canted:
        return np.eye(4)
    tangent = np.array([-np.sin(ang), 0.0, np.cos(ang)])
    return trimesh.transformations.rotation_matrix(np.radians(CANT_DEG), tangent)


def transform_point(M, p):
    v = np.array([p[0], p[1], p[2], 1.0])
    return (M @ v)[:3]


def transform_dir(M, d):
    v = np.array([d[0], d[1], d[2], 0.0])
    return (M @ v)[:3]


print("building solids...")
main_chamber = cyl_along_y(CHAMBER_R, 0.0, DOMAIN_LEN, sections=160)

rp1_stubs = []
rp1_inlet_refs = []   # (point, normal) per element, local frame transformed
for e in elements:
    M_rot = cant_transform(e['ang'], e['canted'])
    s = cyl_along_y(RP1_D / 2.0, STUB_UP, STUB_DOWN, sections=16)
    s.apply_transform(M_rot)
    s.apply_translation([e['x'], 0, e['z']])
    rp1_stubs.append(s)
    p_local = (0, STUB_UP, 0)
    n_local = (0, -1, 0)
    p_world = transform_point(M_rot, p_local) + np.array([e['x'], 0, e['z']])
    n_world = transform_dir(M_rot, n_local)
    rp1_inlet_refs.append((p_world, n_world))

lox_stubs = []
lox_inlet_refs = []   # one per lox element
for e in lox_elements:
    M_rot = cant_transform(e['ang'], e['canted'])
    s = annulus_along_y(CUP_D / 2.0, TUBE_OD / 2.0, STUB_UP, STUB_DOWN, sections=32)
    s.apply_transform(M_rot)
    s.apply_translation([e['x'], 0, e['z']])
    lox_stubs.append(s)
    p_local = (0, STUB_UP, 0)
    n_local = (0, -1, 0)
    p_world = transform_point(M_rot, p_local) + np.array([e['x'], 0, e['z']])
    n_world = transform_dir(M_rot, n_local)
    lox_inlet_refs.append((p_world, n_world, e['ang'], e['canted']))

print(f"unioning {1 + len(rp1_stubs) + len(lox_stubs)} solids...")
domain = main_chamber
for s in rp1_stubs + lox_stubs:
    domain = trimesh.boolean.union([domain, s], engine='manifold')
print("union done. watertight:", domain.is_watertight, "faces:", len(domain.faces))

out_stl = f'{GEOM_DIR}/domain_raw_rev4.stl'
domain.export(out_stl)

centroids = domain.triangles_center
normals = domain.face_normals

is_outlet = (np.abs(centroids[:, 1] - DOMAIN_LEN) < 1e-2) & (normals[:, 1] > 0.9)

# --- classify RP1 (single combined patch): nearest matching inlet ref within radius+normal tol ---
is_rp1 = np.zeros(len(centroids), dtype=bool)
rp1_pts = np.array([r[0] for r in rp1_inlet_refs])
rp1_nrm = np.array([r[1] for r in rp1_inlet_refs])
for i in range(len(rp1_inlet_refs)):
    p, n = rp1_pts[i], rp1_nrm[i]
    d = np.linalg.norm(centroids - p, axis=1)
    align = normals @ n
    is_rp1 |= (d < RP1_D * 0.9) & (align > 0.85)

# --- classify LOX: 36 separate masks, nearest-center wins among candidates ---
lox_pts = np.array([r[0] for r in lox_inlet_refs])
lox_nrm = np.array([r[1] for r in lox_inlet_refs])
lox_candidate = np.zeros(len(centroids), dtype=bool)
lox_owner = -np.ones(len(centroids), dtype=int)
best_dist = np.full(len(centroids), np.inf)
for i in range(len(lox_inlet_refs)):
    p, n = lox_pts[i], lox_nrm[i]
    d = np.linalg.norm(centroids - p, axis=1)
    align = normals @ n
    cand = (d < CUP_D * 0.9) & (align > 0.85)
    lox_candidate |= cand
    better = cand & (d < best_dist)
    lox_owner[better] = i
    best_dist[better] = d[better]

is_wall = ~(is_outlet | is_rp1 | lox_candidate)

print("outlet faces:", is_outlet.sum(), " rp1 faces:", is_rp1.sum(),
      " lox faces (total):", lox_candidate.sum(), " wall faces:", is_wall.sum())

for mask, name in [(is_outlet, 'outlet'), (is_rp1, 'rp1_inlet'), (is_wall, 'walls')]:
    sub = domain.submesh([mask], append=True)
    sub.export(f'{GEOM_DIR}/rev4_{name}.stl')
    print(f"  exported {name}.stl faces={mask.sum()}")

lox_dir = f'{GEOM_DIR}/rev4_lox'
os.makedirs(lox_dir, exist_ok=True)
lox_meta = []
for i in range(len(lox_inlet_refs)):
    mask = (lox_owner == i)
    if mask.sum() == 0:
        print(f"  WARNING lox element {i} got 0 faces")
        continue
    sub = domain.submesh([mask], append=True)
    fname = f'lox_inlet_{i:02d}.stl'
    sub.export(f'{lox_dir}/{fname}')
    p, n, ang, canted = lox_inlet_refs[i]
    lox_meta.append(dict(idx=i, x=float(p[0]) if False else float(lox_elements[i]['x']),
                          z=float(lox_elements[i]['z']), ang=float(ang), canted=bool(canted),
                          origin=[float(v) for v in p], axis=[float(v) for v in -n]))
    print(f"  exported lox_inlet_{i:02d}.stl faces={mask.sum()} canted={canted}")

import json
with open(f'{GEOM_DIR}/rev4_lox_meta.json', 'w') as f:
    json.dump(lox_meta, f, indent=2)

with open(f'{GEOM_DIR}/rev4_counts.txt', 'w') as f:
    f.write(f"n_rp1_active={n_rp1}\nn_lox_active={n_lox}\ncup_d={CUP_D}\ncant_deg={CANT_DEG}\n")

print("DONE rev4")
