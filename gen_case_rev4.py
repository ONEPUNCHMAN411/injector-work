"""
Generate the REV4 OpenFOAM case: REV3's 72-bore pattern + outer-ring cant +
narrowed LOX annulus (J tuning) + per-element LOX swirl via swirlFlowRateInletVelocity
(36 separate lox_inlet_XX patches, each with its own origin/axis from the geometry
metadata, so swirl is local to each coaxial element rather than around the chamber axis).
Own implementation, no external scripts executed.
"""
import sys, os, json, math

CASE_DIR = sys.argv[1]
HOME = os.environ.get('HOME', '/root')
GEOM_DIR = f'{HOME}/injector_work/geom'

Q_LOX = 0.012767  # m3/s, total, matches case_params_coax.py
Q_RP1 = 0.007194
N_RP1 = 72
N_LOX = 36

CUP_D = 7.5      # mm, matches build_domain_rev4.py
TUBE_OD = 6.24
SWIRL_RATIO = 0.6   # V_tangential / V_axial at the annulus mean radius

with open(f'{GEOM_DIR}/rev4_lox_meta.json') as f:
    lox_meta = json.load(f)
assert len(lox_meta) == N_LOX, f"expected {N_LOX} lox elements, got {len(lox_meta)}"

os.makedirs(f'{CASE_DIR}/system', exist_ok=True)
os.makedirs(f'{CASE_DIR}/constant/triSurface', exist_ok=True)
os.makedirs(f'{CASE_DIR}/0.orig', exist_ok=True)


def w(path, content):
    with open(path, 'w') as f:
        f.write(content)


# ---------------- blockMeshDict (unchanged domain) ----------------
w(f'{CASE_DIR}/system/blockMeshDict', """FoamFile
{
    version 2.0; format ascii; class dictionary; object blockMeshDict;
}
scale 0.001;
vertices
(
    (-95 -10 -95)
    (-95 -10  95)
    ( 95 -10  95)
    ( 95 -10 -95)
    (-95 210 -95)
    (-95 210  95)
    ( 95 210  95)
    ( 95 210 -95)
);
blocks ( hex (0 1 2 3 4 5 6 7) (50 58 50) simpleGrading (1 1 1) );
edges ();
boundary
(
    boundingBox
    {
        type patch;
        faces ( (0 1 2 3) (4 5 6 7) (0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7) );
    }
);
mergePatchPairs ();
""")

# ---------------- snappyHexMeshDict ----------------
lox_names = [f'lox_inlet_{m["idx"]:02d}' for m in lox_meta]
geom_lines = ['    walls.stl      { type triSurfaceMesh; name walls;     scale 0.001; }',
              '    outlet.stl     { type triSurfaceMesh; name outlet;    scale 0.001; }',
              '    rp1_inlet.stl  { type triSurfaceMesh; name rp1_inlet; scale 0.001; }']
for nm in lox_names:
    geom_lines.append(f'    {nm}.stl {{ type triSurfaceMesh; name {nm}; scale 0.001; }}')
geom_block = '\n'.join(geom_lines)

refsurf_lines = ['        walls      { level (0 0); }',
                  '        outlet     { level (0 0); }',
                  '        rp1_inlet  { level (4 4); }']
for nm in lox_names:
    refsurf_lines.append(f'        {nm} {{ level (4 4); }}')
refsurf_block = '\n'.join(refsurf_lines)

w(f'{CASE_DIR}/system/snappyHexMeshDict', f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object snappyHexMeshDict;
}}
castellatedMesh true; snap true; addLayers false;

geometry
{{
{geom_block}

    nearFieldBox  {{ type searchableBox; min (-0.095 -0.001 -0.095); max (0.095 0.060 0.095); }}
    jetZoneBox    {{ type searchableBox; min (-0.095 -0.001 -0.095); max (0.095 0.030 0.095); }}
    faceZoneBox   {{ type searchableBox; min (-0.095 -0.001 -0.095); max (0.095 0.004 0.095); }}
}}

castellatedMeshControls
{{
    maxLocalCells 6000000;
    maxGlobalCells 12000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces
    {{
{refsurf_block}
    }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
        nearFieldBox {{ mode inside; levels ((1 1)); }}
        jetZoneBox   {{ mode inside; levels ((2 2)); }}
        faceZoneBox  {{ mode inside; levels ((3 3)); }}
    }}
    locationInMesh (0 0.1 0);
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 5; tolerance 2.0; nSolveIter 100; nRelaxIter 8; nFeatureSnapIter 10;
    implicitFeatureSnap true;
}}
addLayersControls
{{
    relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3;
    minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1;
    nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 30;
    minVol 1e-16; minTetQuality 1e-9; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.05; minVolRatio 0.05; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}
mergeTolerance 1e-6;
""")

# ---------------- fvSchemes / fvSolution / controlDict (same physics as rev2/rev3) ----------------
w(f'{CASE_DIR}/system/fvSchemes', """FoamFile
{ version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")

w(f'{CASE_DIR}/system/fvSolution', """FoamFile
{ version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.05;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl { p 1e-5; U 1e-5; "(k|omega)" 1e-5; }
}
relaxationFactors
{
    equations { U 0.7; "(k|omega)" 0.7; }
    fields { p 0.3; }
}
""")

w(f'{CASE_DIR}/system/controlDict', """FoamFile
{ version 2.0; format ascii; class dictionary; object controlDict; }
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         2000;
deltaT          1;
writeControl    timeStep;
writeInterval   25;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
""")

# ---------------- constant ----------------
w(f'{CASE_DIR}/constant/transportProperties', """FoamFile
{ version 2.0; format ascii; class dictionary; object transportProperties; }
transportModel  Newtonian;
nu              5e-7;
""")
w(f'{CASE_DIR}/constant/turbulenceProperties', """FoamFile
{ version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType RAS;
RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }
""")

# ---------------- flow estimates for turbulence IC + swirl rpm ----------------
RP1_D_M = 0.00242
CUP_D_M = CUP_D / 1000.0
TUBE_OD_M = TUBE_OD / 1000.0

A_rp1 = math.pi * (RP1_D_M / 2) ** 2
A_lox_ann = math.pi * ((CUP_D_M / 2) ** 2 - (TUBE_OD_M / 2) ** 2)

q_rp1_each = Q_RP1 / N_RP1
q_lox_each = Q_LOX / N_LOX

U_rp1_est = q_rp1_each / A_rp1
U_lox_ax = q_lox_each / A_lox_ann

r_mean = (CUP_D_M / 2 + TUBE_OD_M / 2) / 2.0
V_tan = SWIRL_RATIO * U_lox_ax
omega_rad_s = V_tan / r_mean
RPM = omega_rad_s * 60.0 / (2.0 * math.pi)

RHO_LOX = 1141.0
RHO_RP1 = 810.0
J = (RHO_LOX * U_lox_ax ** 2) / (RHO_RP1 * U_rp1_est ** 2)

print(f"U_rp1_est={U_rp1_est:.3f} m/s  U_lox_axial={U_lox_ax:.3f} m/s  V_tan={V_tan:.3f} m/s  "
      f"RPM={RPM:.1f}  J={J:.3f}  (target 1-4)")


def turb_kw(U, D):
    I = 0.05
    k = 1.5 * (I * U) ** 2
    L = 0.07 * D
    Cmu = 0.09
    omega = math.sqrt(k) / (Cmu ** 0.25 * L)
    return k, omega


k_rp1, w_rp1 = turb_kw(U_rp1_est, RP1_D_M)
k_lox, w_lox = turb_kw(math.hypot(U_lox_ax, V_tan), CUP_D_M - TUBE_OD_M)
k_min = min(k_rp1, k_lox)
w_min = min(w_rp1, w_lox)

# ---------------- 0.orig/U ----------------
lox_U_blocks = []
for m in lox_meta:
    name = f'lox_inlet_{m["idx"]:02d}'
    ox, oy, oz = (v / 1000.0 for v in m['origin'])   # mm -> m (mesh is in metres)
    ax, ay, az = m['axis']
    lox_U_blocks.append(f"""    {name}
    {{
        type            swirlFlowRateInletVelocity;
        flowRate        constant {q_lox_each:.8e};
        rpm             constant {RPM:.4f};
        origin          ({ox:.6e} {oy:.6e} {oz:.6e});
        axis            ({ax:.6e} {ay:.6e} {az:.6e});
        value           uniform (0 1 0);
    }}""")
lox_U_block = '\n'.join(lox_U_blocks)

w(f'{CASE_DIR}/0.orig/U', f"""FoamFile
{{ version 2.0; format ascii; class volVectorField; location "0"; object U; }}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    rp1_inlet
    {{
        type            flowRateInletVelocity;
        volumetricFlowRate constant {Q_RP1:.8e};
        value           uniform (0 1 0);
    }}
{lox_U_block}
    outlet
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    walls {{ type noSlip; }}
    boundingBox {{ type noSlip; }}
}}
""")

# ---------------- 0.orig/p ----------------
lox_p_blocks = '\n'.join(f'    lox_inlet_{m["idx"]:02d} {{ type zeroGradient; }}' for m in lox_meta)
w(f'{CASE_DIR}/0.orig/p', f"""FoamFile
{{ version 2.0; format ascii; class volScalarField; location "0"; object p; }}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    rp1_inlet   {{ type zeroGradient; }}
{lox_p_blocks}
    outlet      {{ type totalPressure; p0 uniform 0; value uniform 0; }}
    walls       {{ type zeroGradient; }}
    boundingBox {{ type zeroGradient; }}
}}
""")

# ---------------- 0.orig/k ----------------
lox_k_blocks = '\n'.join(f'    lox_inlet_{m["idx"]:02d} {{ type fixedValue; value uniform {k_lox:.6g}; }}' for m in lox_meta)
w(f'{CASE_DIR}/0.orig/k', f"""FoamFile
{{ version 2.0; format ascii; class volScalarField; location "0"; object k; }}
dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_min:.6g};
boundaryField
{{
    rp1_inlet   {{ type fixedValue; value uniform {k_rp1:.6g}; }}
{lox_k_blocks}
    outlet      {{ type inletOutlet; inletValue uniform {k_min:.6g}; value uniform {k_min:.6g}; }}
    walls       {{ type kqRWallFunction; value uniform {k_min:.6g}; }}
    boundingBox {{ type kqRWallFunction; value uniform {k_min:.6g}; }}
}}
""")

# ---------------- 0.orig/omega ----------------
lox_w_blocks = '\n'.join(f'    lox_inlet_{m["idx"]:02d} {{ type fixedValue; value uniform {w_lox:.6g}; }}' for m in lox_meta)
w(f'{CASE_DIR}/0.orig/omega', f"""FoamFile
{{ version 2.0; format ascii; class volScalarField; location "0"; object omega; }}
dimensions [0 0 -1 0 0 0 0];
internalField uniform {w_min:.6g};
boundaryField
{{
    rp1_inlet   {{ type fixedValue; value uniform {w_rp1:.6g}; }}
{lox_w_blocks}
    outlet      {{ type inletOutlet; inletValue uniform {w_min:.6g}; value uniform {w_min:.6g}; }}
    walls       {{ type omegaWallFunction; value uniform {w_min:.6g}; }}
    boundingBox {{ type omegaWallFunction; value uniform {w_min:.6g}; }}
}}
""")

# ---------------- 0.orig/nut ----------------
lox_nut_blocks = '\n'.join(f'    lox_inlet_{m["idx"]:02d} {{ type calculated; value uniform 0; }}' for m in lox_meta)
w(f'{CASE_DIR}/0.orig/nut', f"""FoamFile
{{ version 2.0; format ascii; class volScalarField; location "0"; object nut; }}
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    rp1_inlet   {{ type calculated; value uniform 0; }}
{lox_nut_blocks}
    outlet      {{ type calculated; value uniform 0; }}
    walls       {{ type nutkWallFunction; value uniform 0; }}
    boundingBox {{ type nutkWallFunction; value uniform 0; }}
}}
""")

print(f"case files written to {CASE_DIR}")
