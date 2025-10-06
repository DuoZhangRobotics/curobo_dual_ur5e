#!/usr/bin/env python3
"""Generate an SRDF from a cuRobo robot YAML config and matching URDF."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - guard rails for missing dep
    raise SystemExit("PyYAML is required to run this script.") from exc


@dataclass(frozen=True)
class JointInfo:
    """Container for URDF joint information."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    mimic: bool


@dataclass
class URDFInfo:
    """Aggregated information extracted from a URDF."""

    robot_name: str
    links: Set[str]
    child_to_joint: Dict[str, JointInfo]
    root_link: Optional[str]
    mimic_joints: Set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an SRDF file by combining collision settings from a cuRobo "
            "robot YAML with link definitions from the corresponding URDF."
        )
    )
    parser.add_argument("yaml", type=Path, help="Path to cuRobo robot YAML config")
    parser.add_argument("urdf", type=Path, help="Path to URDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for the generated SRDF (defaults to stdout)",
    )
    parser.add_argument(
        "--reason",
        default="Configured",
        help="Reason attribute to use for each <disable_collisions> entry",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
    return data


def get_kinematics_cfg(config: dict) -> dict:
    try:
        kin_cfg = config["robot_cfg"]["kinematics"]
    except KeyError as exc:
        raise KeyError("Expected robot_cfg.kinematics section in YAML") from exc
    if not isinstance(kin_cfg, dict):
        raise TypeError("robot_cfg.kinematics must be a mapping")
    return kin_cfg


def extract_collision_settings(kin_cfg: dict) -> tuple[Set[str], Dict[str, List[str]]]:
    collision_links = kin_cfg.get("collision_link_names", [])
    if not isinstance(collision_links, list):
        raise TypeError("robot_cfg.kinematics.collision_link_names must be a list")
    collision_link_set: Set[str] = {str(link) for link in collision_links}

    ignore_map = kin_cfg.get("self_collision_ignore", {})
    if not isinstance(ignore_map, dict):
        raise TypeError("robot_cfg.kinematics.self_collision_ignore must be a mapping")

    normalized_ignore: Dict[str, List[str]] = {}
    for key, value in ignore_map.items():
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "Each entry in self_collision_ignore must be a list or tuple"
            )
        normalized_ignore[str(key)] = [str(item) for item in value]

    return collision_link_set, normalized_ignore


def parse_urdf(path: Path) -> URDFInfo:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"URDF root tag must be 'robot', got '{root.tag}'")

    robot_name = root.get("name") or "unnamed"
    link_names: Set[str] = set()
    child_links: Set[str] = set()
    child_to_joint: Dict[str, JointInfo] = {}
    mimic_joints: Set[str] = set()

    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        if name:
            link_names.add(name)

    for joint_elem in root.findall("joint"):
        name = joint_elem.get("name")
        if not name:
            continue
        joint_type = joint_elem.get("type", "fixed")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent_link = parent_elem.get("link")
        child_link = child_elem.get("link")
        if not parent_link or not child_link:
            continue
        mimic = joint_elem.find("mimic") is not None
        if mimic:
            mimic_joints.add(name)

        info = JointInfo(
            name=name,
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            mimic=mimic,
        )
        child_to_joint[child_link] = info
        child_links.add(child_link)

    root_candidates = link_names - child_links
    root_link = next(iter(root_candidates)) if root_candidates else None

    return URDFInfo(
        robot_name=robot_name,
        links=link_names,
        child_to_joint=child_to_joint,
        root_link=root_link,
        mimic_joints=mimic_joints,
    )


def build_disable_collision_pairs(ignore_map: Dict[str, List[str]]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for link, others in ignore_map.items():
        for other in others:
            if link == other:
                continue
            ordered = tuple(sorted((link, other)))
            pairs.add(ordered)
    return pairs


def filter_pairs_for_urdf(
    pairs: Set[Tuple[str, str]],
    urdf_links: Set[str],
    collision_links: Set[str],
) -> tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    valid: List[Tuple[str, str]] = []
    missing: List[Tuple[str, str]] = []
    for link1, link2 in sorted(pairs):
        if link1 not in collision_links or link2 not in collision_links:
            continue
        valid.append((link1, link2))
        if link1 not in urdf_links or link2 not in urdf_links:
            missing.append((link1, link2))
    return valid, missing


def sanitize_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized or "group"


def resolve_base_link(kin_cfg: dict, urdf_info: URDFInfo) -> Optional[str]:
    base_link = kin_cfg.get("base_link")
    if isinstance(base_link, str) and base_link:
        return base_link
    return urdf_info.root_link


def gather_tip_links(kin_cfg: dict) -> List[str]:
    tips: List[str] = []
    ee_link = kin_cfg.get("ee_link")
    if isinstance(ee_link, str) and ee_link:
        tips.append(ee_link)

    extra_links = kin_cfg.get("link_names")
    if isinstance(extra_links, (list, tuple)):
        for item in extra_links:
            if isinstance(item, str) and item and item not in tips:
                tips.append(item)
    return tips


def compute_joint_chain(
    child_to_joint: Dict[str, JointInfo],
    base_link: str,
    tip_link: str,
) -> Optional[List[str]]:
    chain: List[str] = []
    current = tip_link
    visited: Set[str] = set()
    while current != base_link:
        joint = child_to_joint.get(current)
        if joint is None:
            return None
        chain.append(joint.name)
        current = joint.parent_link
        if current in visited:
            raise ValueError(
                f"Detected cycle in URDF while tracing chain for '{tip_link}'"
            )
        visited.add(current)
    chain.reverse()
    return chain


def build_groups(
    base_link: Optional[str],
    tip_links: Iterable[str],
    child_to_joint: Dict[str, JointInfo],
) -> tuple[List[dict], List[str]]:
    if base_link is None:
        return [], list(tip_links)

    groups: List[dict] = []
    skipped: List[str] = []
    taken_names: Set[str] = set()
    for tip in tip_links:
        chain = compute_joint_chain(child_to_joint, base_link, tip)
        if chain is None:
            skipped.append(tip)
            continue
        group_name = sanitize_name(f"{tip}_group")
        original = group_name
        suffix = 1
        while group_name in taken_names:
            suffix += 1
            group_name = f"{original}_{suffix}"
        taken_names.add(group_name)
        groups.append(
            {
                "name": group_name,
                "base_link": base_link,
                "tip_link": tip,
                "chain_joints": chain,
            }
        )
    return groups, skipped


def extract_joint_defaults(kin_cfg: dict) -> Dict[str, float]:
    cspace = kin_cfg.get("cspace")
    if not isinstance(cspace, dict):
        return {}
    joint_names = cspace.get("joint_names")
    retract = cspace.get("retract_config")
    if not (
        isinstance(joint_names, list)
        and isinstance(retract, list)
        and len(joint_names) == len(retract)
    ):
        return {}
    defaults: Dict[str, float] = {}
    for name, value in zip(joint_names, retract):
        try:
            defaults[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return defaults


def build_group_states(
    groups: List[dict],
    joint_defaults: Dict[str, float],
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    group_states: List[Tuple[str, List[Tuple[str, float]]]] = []
    for group in groups:
        joints_with_values = [
            (joint, joint_defaults[joint])
            for joint in group["chain_joints"]
            if joint in joint_defaults
        ]
        if not joints_with_values:
            continue
        group_states.append((group["name"], joints_with_values))
    return group_states


def collect_passive_joints(kin_cfg: dict, urdf_info: URDFInfo) -> Set[str]:
    passive: Set[str] = set()
    lock_joints = kin_cfg.get("lock_joints", {})
    if isinstance(lock_joints, dict):
        passive.update(str(name) for name in lock_joints.keys())
    passive.update(urdf_info.mimic_joints)
    return passive


def construct_srdf(
    robot_name: str,
    virtual_joint_child: Optional[str],
    groups: List[dict],
    group_states: List[Tuple[str, List[Tuple[str, float]]]],
    passive_joints: Iterable[str],
    disable_pairs: Iterable[Tuple[str, str]],
    reason: str,
) -> str:
    robot_elem = ET.Element("robot", attrib={"name": robot_name})

    if virtual_joint_child:
        ET.SubElement(
            robot_elem,
            "virtual_joint",
            attrib={
                "name": "world_to_base",
                "type": "fixed",
                "parent_frame": "world",
                "child_link": virtual_joint_child,
            },
        )

    for group in groups:
        group_elem = ET.SubElement(robot_elem, "group", attrib={"name": group["name"]})
        ET.SubElement(
            group_elem,
            "chain",
            attrib={
                "base_link": group["base_link"],
                "tip_link": group["tip_link"],
            },
        )

    for group_name, joint_values in group_states:
        state_elem = ET.SubElement(
            robot_elem,
            "group_state",
            attrib={"name": "retract", "group": group_name},
        )
        for joint_name, value in joint_values:
            ET.SubElement(
                state_elem,
                "joint",
                attrib={"name": joint_name, "value": f"{value:.6f}"},
            )

    for joint_name in sorted(set(passive_joints)):
        ET.SubElement(robot_elem, "passive_joint", attrib={"name": joint_name})

    for link1, link2 in disable_pairs:
        ET.SubElement(
            robot_elem,
            "disable_collisions",
            attrib={"link1": link1, "link2": link2, "reason": reason},
        )

    _indent(robot_elem)
    xml_declaration = "<?xml version=\"1.0\"?>\n"
    srdf_bytes = ET.tostring(robot_elem, encoding="unicode")
    return xml_declaration + srdf_bytes


def _indent(elem: ET.Element, level: int = 0) -> None:
    spaces = "  "
    indent_text = "\n" + level * spaces
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_text + spaces
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_text
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent_text


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.yaml)
    kin_cfg = get_kinematics_cfg(config)

    collision_links, ignore_map = extract_collision_settings(kin_cfg)
    urdf_info = parse_urdf(args.urdf)

    base_link = resolve_base_link(kin_cfg, urdf_info)
    tip_links = gather_tip_links(kin_cfg)

    groups, skipped_tips = build_groups(base_link, tip_links, urdf_info.child_to_joint)
    joint_defaults = extract_joint_defaults(kin_cfg)
    group_states = build_group_states(groups, joint_defaults)
    passive_joints = collect_passive_joints(kin_cfg, urdf_info)

    pairs = build_disable_collision_pairs(ignore_map)
    disable_pairs, missing_pairs = filter_pairs_for_urdf(
        pairs, urdf_info.links, collision_links
    )

    if skipped_tips:
        print(
            "Warning: could not build kinematic chains for tips "
            + ", ".join(skipped_tips),
            file=sys.stderr,
        )
    if missing_pairs:
        print(
            "Warning: SRDF references links not present in URDF yet for pairs: "
            + ", ".join(f"({a}, {b})" for a, b in missing_pairs),
            file=sys.stderr,
        )

    srdf_text = construct_srdf(
        robot_name=urdf_info.robot_name,
        virtual_joint_child=base_link,
        groups=groups,
        group_states=group_states,
        passive_joints=passive_joints,
        disable_pairs=disable_pairs,
        reason=args.reason,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(srdf_text, encoding="utf-8")
    else:
        sys.stdout.write(srdf_text)


if __name__ == "__main__":
    main()
