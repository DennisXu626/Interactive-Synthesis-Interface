#!/usr/bin/env python3
"""
parse_scene.py  –  Subtask 1: Parse a SceneSmith scene_XXX.tar into scene_manifest.json

Reads:
  combined_house/house.dmd.yaml   → all object names, SDF paths, world poses
  <object>/model.sdf              → per-object links (GLTF mesh paths) + joints

Writes:
  <output_dir>/scene_manifest.json

scene_manifest.json format:
  {
    "room_frame": {"translation": [x,y,z]},   ← room origin in world
    "objects": [
      {
        "name": "bedroom_nightstand_0",         ← instance name from yaml
        "sdf_path": "room_bedroom/.../model.sdf",  ← relative to extracted root
        "category": "furniture",               ← furniture/manipuland/wall_mounted/ceiling_mounted/room_geometry
        "is_articulated": true,
        "base_link": "E_body_25",              ← link whose world pose is given
        "world_pose": {
          "translation": [x, y, z],            ← in world (Z-up) coordinates
          "rotation_matrix": [[...], [...], [...]]  ← 3x3 SO3
        },
        "links": [
          {"name": "E_body_25", "gltf": "room_bedroom/.../E_body_25_combined.gltf"}
        ],
        "joints": [
          {
            "name": "PrismaticJoint_...",
            "type": "prismatic",               ← prismatic / revolute / fixed
            "parent": "E_body_25",
            "child": "E_drawer_1",
            "axis": [0.0, 1.0, 0.0],          ← in parent-link frame
            "origin": [x, y, z],              ← joint origin in parent-link frame
            "limit": [0.0, 0.2]               ← [lower, upper] in metres or radians
          }
        ]
      }
    ]
  }

Usage:
  python parse_scene.py --tar scene_001.tar --output_dir scenesmith_sample
"""

import os
import sys
import json
import math
import tarfile
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import yaml


# ── YAML: handle Drake's custom !AngleAxis tag ──────────────────────────────

class _AngleAxisLoader(yaml.SafeLoader):
    pass

def _angle_axis_constructor(loader, node):
    data = loader.construct_mapping(node, deep=True)
    return data  # keep as dict: {angle_deg: float, axis: [x,y,z]}

_AngleAxisLoader.add_constructor("!AngleAxis", _angle_axis_constructor)


def angle_axis_to_matrix(angle_deg, axis):
    """Convert AngleAxis rotation to 3x3 rotation matrix (Rodrigues formula)."""
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3)
    axis /= norm
    theta = math.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll-pitch-yaw (radians) to 3x3 rotation matrix (ZYX convention)."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


# ── SDF parsing ──────────────────────────────────────────────────────────────

def _find_model(root):
    """Return the <model> element regardless of whether SDF wraps it in <world>."""
    model = root.find("model")
    if model is not None:
        return model
    model = root.find("world/model")
    if model is not None:
        return model
    # search any depth
    for el in root.iter("model"):
        return el
    return None


def parse_sdf(sdf_content, sdf_rel_dir):
    """
    Parse SDF XML content.

    Returns:
        links:  list of {
                  "name":          unique visual ID (e.g. "base_link/visual"),
                  "fk_link":       SDF link name to use for FK lookup,
                  "gltf":          path relative to extracted root (or None),
                  "mesh_scale":    uniform scale factor,
                  "visual_offset": [x, y, z, rx, ry, rz] pose of this visual
                                   in the link frame (SDF Z-up)
                }
        joints: list of {"name", "type", "parent", "child",
                         "axis": [x,y,z], "origin": [x,y,z], "limit": [lo,hi]}
        base_link: name of the first SDF link (used as pose anchor)
    """
    root = ET.fromstring(sdf_content)
    model = _find_model(root)
    if model is None:
        return [], [], None

    links = []
    for link in model.findall("link"):
        lname = link.get("name")
        visuals = link.findall("visual")
        if not visuals:
            # Link with no visual — still register as a placeholder
            links.append({"name": lname, "fk_link": lname, "gltf": None,
                          "mesh_scale": 1.0, "visual_offset": [0,0,0,0,0,0]})
            continue

        for visual in visuals:
            vname = visual.get("name", "visual")
            unique_name = f"{lname}/{vname}" if vname != lname else lname

            uri_el   = visual.find("geometry/mesh/uri")
            scale_el = visual.find("geometry/mesh/scale")

            mesh_scale = 1.0
            if scale_el is not None:
                vals = [float(v) for v in scale_el.text.split()]
                mesh_scale = vals[0]  # assume uniform scale

            # Visual pose within the link frame (SDF Z-up)
            pose_el = visual.find("pose")
            visual_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if pose_el is not None and pose_el.text:
                visual_offset = [float(v) for v in pose_el.text.split()]

            if uri_el is not None:
                gltf_rel = os.path.join(sdf_rel_dir, uri_el.text).replace("\\", "/")
            else:
                gltf_rel = None

            links.append({"name": unique_name, "fk_link": lname, "gltf": gltf_rel,
                          "mesh_scale": mesh_scale, "visual_offset": visual_offset})

    joints = []
    for joint in model.findall("joint"):
        jname = joint.get("name")
        jtype = joint.get("type", "fixed")
        if jtype == "fixed":
            continue  # skip fixed joints — child is rigidly attached to parent
        parent_el = joint.find("parent")
        child_el  = joint.find("child")
        parent = parent_el.text.strip() if parent_el is not None else ""
        child  = child_el.text.strip()  if child_el  is not None else ""

        # Axis vector (in joint frame; joint frame = parent frame for rpy=0)
        axis_el = joint.find("axis/xyz")
        axis = [float(v) for v in axis_el.text.split()] if axis_el is not None else [0, 0, 1]

        # Joint limits
        lo_el = joint.find("axis/limit/lower")
        hi_el = joint.find("axis/limit/upper")
        lo = float(lo_el.text) if lo_el is not None else 0.0
        hi = float(hi_el.text) if hi_el is not None else 0.0

        # Joint pose in parent-link frame: "x y z roll pitch yaw"
        pose_el = joint.find("pose")
        if pose_el is not None:
            vals = [float(v) for v in pose_el.text.split()]
            origin = vals[:3]
            rpy    = vals[3:]
        else:
            origin = [0.0, 0.0, 0.0]
            rpy    = [0.0, 0.0, 0.0]

        # Transform axis from joint frame to parent-link frame
        R_joint_in_parent = rpy_to_matrix(*rpy)
        axis_in_parent = (R_joint_in_parent @ np.array(axis)).tolist()

        joints.append({
            "name":   jname,
            "type":   jtype,
            "parent": parent,
            "child":  child,
            "axis":   axis_in_parent,
            "origin": origin,
            "limit":  [lo, hi],
        })

    # base_link is the SDF link name (fk_link) of the first link entry
    base_link = links[0]["fk_link"] if links else None
    return links, joints, base_link


# ── YAML scene layout parsing ────────────────────────────────────────────────

def _infer_category(sdf_rel_path):
    p = sdf_rel_path.replace("\\", "/")
    if "furniture"       in p: return "furniture"
    if "manipuland"      in p: return "manipuland"
    if "wall_mounted"    in p: return "wall_mounted"
    if "ceiling_mounted" in p: return "ceiling_mounted"
    if "room_geometry"   in p: return "room_geometry"
    return "other"


def parse_yaml_layout(yaml_content):
    """
    Parse house.dmd.yaml directives.

    Returns:
        room_translation: [x, y, z]  (room frame offset in world)
        models: list of dicts:
          {name, sdf_package_path, base_link, translation, rotation_matrix, is_welded}
    """
    data = yaml.load(yaml_content, Loader=_AngleAxisLoader)
    directives = data.get("directives", [])

    # 1. Find room frame translation
    room_translation = [0.0, 0.0, 0.0]
    for d in directives:
        if "add_frame" in d:
            fr = d["add_frame"]
            if "room_" in fr.get("name", ""):
                t = fr.get("X_PF", {}).get("translation", [0, 0, 0])
                room_translation = [float(v) for v in t]
                break

    # 2. Collect all models
    # We need add_model + either add_weld (static) or default_free_body_pose (dynamic)
    # Process directives in order; weld immediately follows the model it welds

    # Index models by name for quick lookup
    model_map = {}  # name -> partial dict
    for d in directives:
        if "add_model" in d:
            am = d["add_model"]
            name = am["name"]
            pkg_path = am.get("file", "")
            dfbp = am.get("default_free_body_pose", None)

            trans = [0.0, 0.0, 0.0]
            rot   = np.eye(3)
            blink = None
            is_welded = False

            if dfbp is not None:
                # dynamic object: pose given as default_free_body_pose
                # dfbp = {link_name: {translation: [...], rotation: !AngleAxis{...}, base_frame: ...}}
                blink = list(dfbp.keys())[0]
                pose_info = dfbp[blink]
                t_val = pose_info.get("translation", [0, 0, 0])
                trans = [float(v) for v in t_val]
                r_val = pose_info.get("rotation", None)
                if r_val is not None and isinstance(r_val, dict):
                    rot = angle_axis_to_matrix(r_val["angle_deg"], r_val["axis"])

            model_map[name] = {
                "name": name,
                "sdf_package_path": pkg_path,
                "base_link": blink,
                "translation": trans,
                "rotation_matrix": rot.tolist(),
                "is_welded": is_welded,
            }

        if "add_weld" in d:
            aw = d["add_weld"]
            # child looks like "model_name::link_name"
            child_str = aw.get("child", "")
            if "::" in child_str:
                mname, lname = child_str.split("::", 1)
            else:
                mname, lname = child_str, None

            if mname in model_map:
                model_map[mname]["is_welded"] = True
                if lname and model_map[mname]["base_link"] is None:
                    model_map[mname]["base_link"] = lname
                # Pose comes from X_PC
                xpc = aw.get("X_PC", None)
                if xpc:
                    t_val = xpc.get("translation", [0, 0, 0])
                    model_map[mname]["translation"] = [float(v) for v in t_val]
                    r_val = xpc.get("rotation", None)
                    if r_val is not None and isinstance(r_val, dict):
                        rot = angle_axis_to_matrix(r_val["angle_deg"], r_val["axis"])
                        model_map[mname]["rotation_matrix"] = rot.tolist()

    models = list(model_map.values())
    return room_translation, models


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_sdf_path(package_path, extracted_root):
    """
    Convert 'package://scene/room_bedroom/.../model.sdf'
    to a path relative to the extracted root.
    """
    # Strip 'package://scene/' prefix
    rel = package_path.replace("package://scene/", "").replace("package://scene\\", "")
    return rel  # relative to extracted root


# ── Main pipeline ─────────────────────────────────────────────────────────────

def parse_scene(tar_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[parse_scene] Extracting {tar_path} ...")
    extract_dir = os.path.join(output_dir, "scene_extracted")
    with tarfile.open(tar_path) as t:
        t.extractall(extract_dir)
    print(f"[parse_scene] Extracted to {extract_dir}")

    # Read house.dmd.yaml
    yaml_path = os.path.join(extract_dir, "combined_house", "house.dmd.yaml")
    if not os.path.exists(yaml_path):
        # try alternate
        yaml_path = os.path.join(extract_dir, "combined_house", "house_furniture_welded.dmd.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    room_translation, models = parse_yaml_layout(yaml_content)
    print(f"[parse_scene] Room frame offset: {room_translation}")
    print(f"[parse_scene] Found {len(models)} models in scene layout")

    # Apply room frame offset to all object world translations
    room_t = np.array(room_translation)
    for m in models:
        # world_translation = room_frame + R_room * local_translation
        # room frame has no rotation (it's just a translation offset in house.dmd.yaml)
        m["world_translation"] = (room_t + np.array(m["translation"])).tolist()
        m["world_rotation_matrix"] = m["rotation_matrix"]  # room frame is axis-aligned

    # Parse each model's SDF for links and joints
    objects = []
    for m in models:
        sdf_rel = resolve_sdf_path(m["sdf_package_path"], extract_dir)
        sdf_abs = os.path.join(extract_dir, sdf_rel)
        sdf_rel_dir = os.path.dirname(sdf_rel).replace("\\", "/")
        category = _infer_category(sdf_rel)

        if not os.path.exists(sdf_abs):
            print(f"  [WARN] SDF not found: {sdf_abs}")
            continue

        with open(sdf_abs, "r", encoding="utf-8") as f:
            sdf_content = f.read()

        links, joints, auto_base = parse_sdf(sdf_content, sdf_rel_dir)
        base_link = m["base_link"] or auto_base

        obj = {
            "name":          m["name"],
            "category":      category,
            "sdf_path":      sdf_rel,
            "is_articulated": len(joints) > 0,
            "is_welded":     m["is_welded"],
            "base_link":     base_link,
            "world_pose": {
                "translation":     m["world_translation"],
                "rotation_matrix": m["world_rotation_matrix"],
            },
            "links":  links,
            "joints": joints,
        }
        objects.append(obj)

    n_art = sum(1 for o in objects if o["is_articulated"])
    print(f"[parse_scene] Parsed {len(objects)} objects ({n_art} articulated)")

    manifest = {
        "source_tar":       os.path.basename(tar_path),
        "extract_dir":      os.path.abspath(extract_dir).replace("\\", "/"),
        "room_frame_translation": room_translation,
        "objects":          objects,
    }

    manifest_path = os.path.join(output_dir, "scene_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[parse_scene] Saved manifest → {manifest_path}")
    return manifest_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse a SceneSmith .tar into scene_manifest.json")
    parser.add_argument("--tar",        default="D:/4YP/singapo/Viser_trial/scene_001.tar",
                        help="Path to scene_XXX.tar")
    parser.add_argument("--output_dir", default="D:/4YP/singapo/Viser_trial/scenesmith_sample",
                        help="Directory to extract into and write manifest")
    args = parser.parse_args()

    manifest_path = parse_scene(args.tar, args.output_dir)

    # Print a quick summary
    with open(manifest_path) as f:
        data = json.load(f)

    print()
    print("=" * 60)
    print("Scene Summary")
    print("=" * 60)
    for cat in ["room_geometry", "furniture", "wall_mounted", "ceiling_mounted", "manipuland", "other"]:
        objs = [o for o in data["objects"] if o["category"] == cat]
        if not objs:
            continue
        print(f"\n  [{cat}]  ({len(objs)} objects)")
        for o in objs:
            art = " [ARTICULATED]" if o["is_articulated"] else ""
            n_links = len(o["links"])
            print(f"    {o['name']}  ({n_links} links){art}")
            for j in o["joints"]:
                print(f"      joint: {j['type']}  {j['parent']}→{j['child']}  "
                      f"axis={[round(v,3) for v in j['axis']]}  limit={j['limit']}")


if __name__ == "__main__":
    main()
