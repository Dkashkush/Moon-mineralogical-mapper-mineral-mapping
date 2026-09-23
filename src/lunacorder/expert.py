"""Expert-system definitions: which features identify which material.

Tetracorder's accuracy comes less from its fitting maths than from its curated
rules: which absorption features are diagnostic for each material, where their
continua sit, and what thresholds separate a detection from noise. Here those
rules live in a YAML file (see ``data/lunar_m3.yaml``) so they can be read,
cited, reviewed and extended without touching code.

YAML layout::

    name: lunar-m3
    wavelength_range_nm: [540, 2500]
    groups:
      mafic: "Fe2+ crystal-field absorptions"
    materials:
      - name: Olivine
        group: mafic
        color: "#2ca25f"
        references: ["^Olivine_"]          # regex on library spectrum names
        exclude: []                        # regex to drop matching references
        features:
          - name: 1 um composite band
            left: [690, 760]               # continuum interval (nm)
            right: [1540, 1660]
        absent:                            # features that must NOT be present
          - name: 2 um pyroxene band
            left: [1540, 1660]
            right: [2380, 2500]
            max_depth: 0.03
        min_fit: 0.85
        min_depth: 0.03
        min_snr: 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml


@dataclass
class FeatureDef:
    name: str
    left: tuple[float, float]
    right: tuple[float, float]
    min_fit: float = 0.0


@dataclass
class AbsentDef:
    name: str
    left: tuple[float, float]
    right: tuple[float, float]
    max_depth: float


@dataclass
class MaterialDef:
    name: str
    group: str
    references: list[str]
    features: list[FeatureDef]
    exclude: list[str] = field(default_factory=list)
    absent: list[AbsentDef] = field(default_factory=list)
    min_fit: float = 0.8
    min_depth: float = 0.02
    min_snr: float = 0.0
    color: str = "#888888"
    notes: str = ""


@dataclass
class ExpertSystem:
    name: str
    wavelength_range_nm: tuple[float, float]
    groups: dict[str, str]
    materials: list[MaterialDef]

    def materials_in(self, group: str) -> list[int]:
        return [i for i, m in enumerate(self.materials) if m.group == group]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExpertSystem":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def builtin(cls, name: str = "lunar_m3") -> "ExpertSystem":
        text = resources.files("lunacorder.data").joinpath(f"{name}.yaml").read_text()
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExpertSystem":
        materials = []
        for m in cfg["materials"]:
            if m["group"] not in cfg["groups"]:
                raise ValueError(f"Material {m['name']} uses undefined group {m['group']}")
            if not m.get("features"):
                raise ValueError(f"Material {m['name']} needs at least one feature")
            materials.append(MaterialDef(
                name=m["name"],
                group=m["group"],
                references=list(m["references"]),
                exclude=list(m.get("exclude", [])),
                features=[FeatureDef(name=f["name"], left=_pair(f["left"]), right=_pair(f["right"]),
                                     min_fit=float(f.get("min_fit", 0.0)))
                          for f in m["features"]],
                absent=[AbsentDef(name=a["name"], left=_pair(a["left"]), right=_pair(a["right"]),
                                  max_depth=float(a["max_depth"]))
                        for a in m.get("absent", [])],
                min_fit=float(m.get("min_fit", 0.8)),
                min_depth=float(m.get("min_depth", 0.02)),
                min_snr=float(m.get("min_snr", 0.0)),
                color=str(m.get("color", "#888888")),
                notes=str(m.get("notes", "")),
            ))
        return cls(
            name=str(cfg.get("name", "unnamed")),
            wavelength_range_nm=_pair(cfg.get("wavelength_range_nm", (0, 1e9))),
            groups=dict(cfg["groups"]),
            materials=materials,
        )


def _pair(values) -> tuple[float, float]:
    a, b = (float(v) for v in values)
    if b < a:
        raise ValueError(f"Interval {values} is reversed")
    return a, b
