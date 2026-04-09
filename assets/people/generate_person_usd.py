#!/usr/bin/env python3
"""Generate a simple human-shaped USD mesh (head sphere + body cylinder + leg cylinders).

This creates a purely visual USD with no physics, intended to be referenced as
a child prim under the kinematic capsule rigid body.

The character stands 1.8 m tall with feet at z=0:
  - Legs:   two cylinders from z=0.0 to z=0.7
  - Body:   cylinder from z=0.7 to z=1.5
  - Head:   sphere centered at z=1.65 (radius 0.15)

Run this OUTSIDE of IsaacSim (plain USD Python environment):
    python generate_person_usd.py
"""

from pxr import Usd, UsdGeom, Gf, Sdf

OUTPUT_PATH = "/home/azureuser/Desktop/IsaacLab/assets/people/person.usda"

stage = Usd.Stage.CreateNew(OUTPUT_PATH)
stage.SetMetadata("upAxis", "Z")
stage.SetMetadata("metersPerUnit", 1.0)

root = UsdGeom.Xform.Define(stage, "/Person")
stage.SetDefaultPrim(root.GetPrim())

# ---- BODY (main torso cylinder) ----
body = UsdGeom.Cylinder.Define(stage, "/Person/Body")
body.GetRadiusAttr().Set(0.18)
body.GetHeightAttr().Set(0.80)
body.GetAxisAttr().Set("Z")
# Center at z=1.1, so bottom at z=0.7, top at z=1.5
UsdGeom.XformCommonAPI(body).SetTranslate(Gf.Vec3d(0.0, 0.0, 1.1))
body.GetDisplayColorAttr().Set([(0.4, 0.3, 0.8)])  # purple-ish shirt

# ---- HEAD (sphere) ----
head = UsdGeom.Sphere.Define(stage, "/Person/Head")
head.GetRadiusAttr().Set(0.15)
# Center at z=1.65
UsdGeom.XformCommonAPI(head).SetTranslate(Gf.Vec3d(0.0, 0.0, 1.65))
head.GetDisplayColorAttr().Set([(0.9, 0.75, 0.60)])  # skin tone

# ---- LEFT LEG ----
left_leg = UsdGeom.Cylinder.Define(stage, "/Person/LeftLeg")
left_leg.GetRadiusAttr().Set(0.07)
left_leg.GetHeightAttr().Set(0.70)
left_leg.GetAxisAttr().Set("Z")
# Center at z=0.35, offset in X
UsdGeom.XformCommonAPI(left_leg).SetTranslate(Gf.Vec3d(-0.09, 0.0, 0.35))
left_leg.GetDisplayColorAttr().Set([(0.2, 0.2, 0.5)])  # dark trousers

# ---- RIGHT LEG ----
right_leg = UsdGeom.Cylinder.Define(stage, "/Person/RightLeg")
right_leg.GetRadiusAttr().Set(0.07)
right_leg.GetHeightAttr().Set(0.70)
right_leg.GetAxisAttr().Set("Z")
UsdGeom.XformCommonAPI(right_leg).SetTranslate(Gf.Vec3d(0.09, 0.0, 0.35))
right_leg.GetDisplayColorAttr().Set([(0.2, 0.2, 0.5)])

# ---- LEFT ARM ----
left_arm = UsdGeom.Cylinder.Define(stage, "/Person/LeftArm")
left_arm.GetRadiusAttr().Set(0.055)
left_arm.GetHeightAttr().Set(0.55)
left_arm.GetAxisAttr().Set("X")
# Shoulder height ~1.35, extending outward in X
UsdGeom.XformCommonAPI(left_arm).SetTranslate(Gf.Vec3d(-0.30, 0.0, 1.30))
left_arm.GetDisplayColorAttr().Set([(0.4, 0.3, 0.8)])

# ---- RIGHT ARM ----
right_arm = UsdGeom.Cylinder.Define(stage, "/Person/RightArm")
right_arm.GetRadiusAttr().Set(0.055)
right_arm.GetHeightAttr().Set(0.55)
right_arm.GetAxisAttr().Set("X")
UsdGeom.XformCommonAPI(right_arm).SetTranslate(Gf.Vec3d(0.30, 0.0, 1.30))
right_arm.GetDisplayColorAttr().Set([(0.4, 0.3, 0.8)])

stage.GetRootLayer().Save()
print(f"Person USD written to: {OUTPUT_PATH}")
