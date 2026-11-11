#!/usr/bin/env python3
"""Read-only, simulator-free audit of checkpoint and composed-evaluation hands."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def model_pair(side):
    prefix = "right" if side == "r" else "left"
    training = ROOT / "rsc/mano_double" / (side + "hand_mano.urdf")
    demo = ROOT / "rsc/mano_double" / (side + "hand_mano_demo.urdf")
    a = ElementTree.parse(str(training)).getroot()
    b = ElementTree.parse(str(demo)).getroot()
    differences = []

    def compare(x, y, path=""):
        assert x.tag == y.tag
        path += "/" + x.tag + (":" + x.attrib["name"] if "name" in x.attrib else "")
        if x.attrib != y.attrib:
            differences.append({"xml_path": path, "training_attributes": dict(x.attrib),
                                "demo_attributes": dict(y.attrib)})
        if (x.text or "").strip() != (y.text or "").strip():
            differences.append({"xml_path": path, "training_text": x.text, "demo_text": y.text})
        assert len(x) == len(y)
        for xx, yy in zip(x, y):
            compare(xx, yy, path)

    compare(a, b)
    expected_joints = [prefix + "_wrist_0" + axis for axis in "xyz"]
    assert len(differences) == 3
    assert all(d["xml_path"].endswith("/joint:" + joint + "/limit")
               for d, joint in zip(differences, expected_joints))
    normalized = copy.deepcopy(a)
    for joint in normalized.findall("joint"):
        if joint.attrib["name"] in expected_joints:
            joint.find("limit").attrib.update(lower="-2.8", upper="2.8")
    identical_other_xml = ElementTree.tostring(normalized) == ElementTree.tostring(b)
    assert identical_other_xml
    relative_demo = str(demo.relative_to(ROOT))
    first_commit = git("log", "--reverse", "--format=%H", "--", relative_demo).splitlines()[0]
    historic = subprocess.check_output(["git", "show", first_commit + ":" + relative_demo], cwd=str(ROOT))
    historic_root = ElementTree.fromstring(historic)
    historic_limits = {j.attrib["name"]: dict(j.find("limit").attrib)
                       for j in historic_root.findall("joint") if j.attrib["name"] in expected_joints}
    return {"training_path": str(training.relative_to(ROOT)), "demo_path": relative_demo,
            "differences": differences, "difference_count": len(differences),
            "identical_other_xml": identical_other_xml,
            "unchanged_categories": ["link topology", "mass", "inertia", "collision geometry",
                                     "visual geometry", "joint origin/axis", "damping/friction",
                                     "effort/velocity limits", "finger joint limits", "rotational wrist limits"],
            "first_demo_commit": first_commit,
            "first_demo_commit_subject": git("show", "-s", "--format=%s", first_commit),
            "first_demo_blob_sha256": hashlib.sha256(historic).hexdigest(),
            "first_demo_wrist_limits": historic_limits,
            "current_demo_matches_first_blob": demo.read_bytes() == historic}


def config_selection(relative):
    path = ROOT / relative
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        match = re.match(r'\s*(hand_model_[rl])\s*:\s*"([^"]+)"', line)
        if match:
            values[match.group(1)] = {"value": match.group(2), "line": number}
    return {"path": relative, "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_configs = [
        "raisimGymTorch/data_all/general_two/2023-05-15-10-18-54/cfg_reg.yaml",
        "raisimGymTorch/data_all/multi_obj_arti/2023-05-14-16-30-13/cfg_reg.yaml"]
    pairs = {side: model_pair(side) for side in ("r", "l")}
    sources = checkpoint_configs + [
        "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/compose_eval.py",
        "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/ketchup_eval.py",
        "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/Environment.hpp",
        str(Path(__file__).resolve().relative_to(ROOT))]
    for pair in pairs.values():
        sources.extend([pair["training_path"], pair["demo_path"]])
    result = {
        "schema_version": 1, "git_commit": git("rev-parse", "HEAD"),
        "method": "ElementTree recursive attribute/text comparison; no simulator or physics execution",
        "hands": pairs, "checkpoint_configurations": [config_selection(p) for p in checkpoint_configs],
        "public_evaluator_selection": {
            "compose_eval.py": {"right_line": 148, "left_line": 149},
            "ketchup_eval.py": {"right_line": 149, "left_line": 150},
            "right_default": "rhand_mano_demo.urdf", "left_default": "lhand_mano_demo.urdf",
            "mesh_collision_flag": "selects *_mano_meshcoll.urdf instead"},
        "control_effect": {
            "source": "raisimGymTorch/raisimGymTorch/env/envs/compose_eval/Environment.hpp",
            "target_clipping_lines": [1020, 1029],
            "fact": "PD targets are clipped to URDF joint limits; symmetric limits retain identical zero centers",
            "interpretation": "Limit expansion can affect large wrist excursions; no causal performance claim established"},
        "conclusion": "Training/demo URDFs differ only in 3 wrist translation lower/upper limits per hand: +/-0.8m vs +/-2.8m. All other parsed XML is identical. Demo variant is present in the first source commit.",
        "source_sha256": {p: sha256(ROOT / p) for p in sources}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("Saved hand model audit:", args.output)


if __name__ == "__main__":
    main()
