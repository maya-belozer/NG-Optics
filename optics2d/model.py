from __future__ import annotations

from .i18n import tr

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4


ELEMENT_KINDS = (
    "horn",
    "lens",
    "plane_mirror",
    "curved_mirror",
    "elliptical_mirror",
    "cryostat",
    "ruler",
    "target",
    "block",
)


@dataclass
class FrequencyChannel:
    frequency_ghz: float = 243.0
    enabled: bool = True
    uid: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class ObjectLink:
    source_uid: str
    target_uid: str
    enabled: bool = True
    uid: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ObjectLink":
        return cls(**value)


@dataclass
class OpticalElement:
    """One optical object in the global XY plane (all lengths are in mm)."""

    kind: str
    name: str
    x: float
    y: float
    angle_deg: float = 0.0
    aperture: float = 36.0
    focal_length: float = 100.0
    target_x: float | None = None
    target_y: float | None = None
    frequency_ghz: float = 243.0
    frequency_channels: list[FrequencyChannel] = field(default_factory=list)
    waist_radius: float = 2.0
    pcl_depth: float = 3.23
    horn_length: float = 27.5
    block_length: float = 40.0
    focus1_x: float | None = None
    focus1_y: float | None = None
    focus2_x: float | None = None
    focus2_y: float | None = None
    radius: float = 120.0
    inner_radius: float = 95.0
    inner_offset_x: float = 25.0
    inner_offset_y: float = 0.0
    window_count: int = 1
    position_mode: str = "coordinates"
    relative_distance: float = 100.0
    relative_angle_deg: float = 0.0
    enabled: bool = True
    uid: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if self.kind not in ELEMENT_KINDS:
            raise ValueError(f"Unknown element kind: {self.kind}")
        if self.position_mode not in ("coordinates", "relative"):
            self.position_mode = "coordinates"
        converted: list[FrequencyChannel] = []
        for channel in self.frequency_channels:
            converted.append(
                channel if isinstance(channel, FrequencyChannel) else FrequencyChannel(**channel)
            )
        self.frequency_channels = converted
        if self.kind == "horn" and not self.frequency_channels:
            self.frequency_channels = [FrequencyChannel(self.frequency_ghz, True)]

    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle_deg)

    @property
    def waveguide_x(self) -> float:
        """X coordinate of a horn's waveguide end; horn x/y is the PCL."""
        back = max(self.horn_length - self.pcl_depth, 0.0)
        return self.x - back * math.cos(self.angle_rad)

    @property
    def waveguide_y(self) -> float:
        """Y coordinate of a horn's waveguide end; horn x/y is the PCL."""
        back = max(self.horn_length - self.pcl_depth, 0.0)
        return self.y - back * math.sin(self.angle_rad)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OpticalElement":
        known = cls.__dataclass_fields__.keys()
        return cls(**{key: val for key, val in value.items() if key in known})

    def active_frequencies(self) -> list[FrequencyChannel]:
        if self.kind != "horn":
            return []
        return [channel for channel in self.frequency_channels if channel.enabled and channel.frequency_ghz > 0]


@dataclass
class OpticalSystem:
    elements: list[OpticalElement] = field(default_factory=list)
    max_path_mm: float = 600.0
    beam_scale: float = 2.0
    render_beam_bounds: bool = True
    simulate_boundary_reflections: bool = False
    link_horn_to_ellipse: bool = True
    link_plane_to_ellipse: bool = True
    links: list[ObjectLink] = field(default_factory=list)
    diagram_positions: dict[str, list[float]] = field(default_factory=dict)

    @property
    def horn(self) -> OpticalElement | None:
        return next((item for item in self.elements if item.kind == "horn" and item.enabled), None)

    @property
    def horns(self) -> list[OpticalElement]:
        return [item for item in self.elements if item.kind == "horn" and item.enabled]

    def element(self, uid: str) -> OpticalElement | None:
        return next((item for item in self.elements if item.uid == uid), None)

    def outgoing(self, uid: str) -> list[OpticalElement]:
        targets = [link.target_uid for link in self.links if link.enabled and link.source_uid == uid]
        return [item for target in targets if (item := self.element(target)) is not None]

    def incoming(self, uid: str) -> list[OpticalElement]:
        sources = [link.source_uid for link in self.links if link.enabled and link.target_uid == uid]
        return [item for source in sources if (item := self.element(source)) is not None]

    def connect(self, source_uid: str, target_uid: str) -> ObjectLink:
        if source_uid == target_uid:
            raise ValueError(tr('An object cannot be connected to itself'))
        source = self.element(source_uid)
        target = self.element(target_uid)
        if source is None or target is None:
            raise ValueError(tr('Both connected objects must exist'))
        if "block" in (source.kind, target.kind):
            if {source.kind, target.kind} != {"block", "horn"}:
                raise ValueError(tr('A block can only connect to the waveguide end of a horn'))
            if source.kind == "block" and any(
                link.enabled and link.source_uid == source.uid for link in self.links
            ):
                raise ValueError(tr('A horn is already connected to the output end of the block'))
            if target.kind == "block" and any(
                link.enabled and link.target_uid == target.uid for link in self.links
            ):
                raise ValueError(tr('A horn is already connected to the input end of the block'))
        # A graph port may fan in/out and exact parallel edges are intentional.
        link = ObjectLink(source_uid, target_uid)
        self.links.append(link)
        return link

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "optics2d-v1",
            "max_path_mm": self.max_path_mm,
            "beam_scale": self.beam_scale,
            "render_beam_bounds": self.render_beam_bounds,
            "simulate_boundary_reflections": self.simulate_boundary_reflections,
            "link_horn_to_ellipse": self.link_horn_to_ellipse,
            "link_plane_to_ellipse": self.link_plane_to_ellipse,
            "elements": [item.to_dict() for item in self.elements],
            "links": [link.to_dict() for link in self.links],
            "diagram_positions": self.diagram_positions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpticalSystem":
        if payload.get("format") != "optics2d-v1":
            raise ValueError("Unsupported optical-system data")
        return cls(
            elements=[OpticalElement.from_dict(item) for item in payload["elements"]],
            max_path_mm=float(payload.get("max_path_mm", 600.0)),
            beam_scale=float(payload.get("beam_scale", 2.0)),
            render_beam_bounds=bool(payload.get("render_beam_bounds", True)),
            simulate_boundary_reflections=bool(payload.get("simulate_boundary_reflections", False)),
            link_horn_to_ellipse=bool(payload.get("link_horn_to_ellipse", True)),
            link_plane_to_ellipse=bool(payload.get("link_plane_to_ellipse", True)),
            links=[ObjectLink.from_dict(item) for item in payload.get("links", [])],
            diagram_positions={key: list(value) for key, value in payload.get("diagram_positions", {}).items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OpticalSystem":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _mirror_angle(incoming: tuple[float, float], outgoing: tuple[float, float]) -> float:
    """Tangent angle of a plane mirror that maps incoming direction to outgoing."""
    ix, iy = incoming
    ox, oy = outgoing
    il = math.hypot(ix, iy)
    ol = math.hypot(ox, oy)
    ix, iy = ix / il, iy / il
    ox, oy = ox / ol, oy / ol
    nx, ny = ix - ox, iy - oy
    nl = math.hypot(nx, ny)
    if nl < 1e-12:
        return math.degrees(math.atan2(iy, ix)) + 90.0
    nx, ny = nx / nl, ny / nl
    return math.degrees(math.atan2(nx, -ny))


def default_band6_system() -> OpticalSystem:
    """Editable preset reconstructed from Optics_B6_wetCyo_V2.nb."""
    pcl = (10.0, -10.0)
    m1 = (23.2, -59.262)
    m2 = (60.0, 0.0)
    d01 = (m1[0] - pcl[0], m1[1] - pcl[1])
    d12 = (m2[0] - m1[0], m2[1] - m1[1])
    d2 = math.hypot(*d12) + 150.0
    d12_unit = (d12[0] / math.hypot(*d12), d12[1] / math.hypot(*d12))
    f2 = (m1[0] + d2 * d12_unit[0], m1[1] + d2 * d12_unit[1])
    horn_angle = math.degrees(math.atan2(d01[1], d01[0]))
    m1_angle = _mirror_angle(d01, d12)
    target = (210.0, 0.0)
    m2_angle = _mirror_angle(d12, (target[0] - m2[0], target[1] - m2[1]))
    system = OpticalSystem(
        elements=[
            OpticalElement(
                "cryostat", tr('Band 6 cryostat'), 0.0, 0.0,
                radius=120.0, inner_radius=95.0,
                inner_offset_x=25.0, inner_offset_y=0.0,
            ),
            OpticalElement(
                "horn", tr('Band 6 horn'), *pcl, angle_deg=horn_angle,
                aperture=6.0, frequency_ghz=243.0, waist_radius=2.0,
                pcl_depth=3.23, horn_length=27.5,
            ),
            OpticalElement(
                "elliptical_mirror", tr('Elliptical mirror M1'), *m1,
                angle_deg=m1_angle, aperture=55.0, focal_length=0.0,
                focus1_x=pcl[0], focus1_y=pcl[1],
                focus2_x=f2[0], focus2_y=f2[1],
                target_x=f2[0], target_y=f2[1],
            ),
            OpticalElement(
                "plane_mirror", tr('Plane mirror M2'), *m2,
                angle_deg=m2_angle, aperture=70.0,
                target_x=target[0], target_y=target[1],
            ),
            OpticalElement(
                "target", tr('Target M2'), *target,
                position_mode="relative", relative_distance=150.0,
                relative_angle_deg=0.0,
            ),
        ],
        max_path_mm=420.0,
        beam_scale=2.0,
        render_beam_bounds=True,
        simulate_boundary_reflections=False,
        link_horn_to_ellipse=True,
        link_plane_to_ellipse=True,
    )
    horn = next(item for item in system.elements if item.kind == "horn")
    ellipse = next(item for item in system.elements if item.kind == "elliptical_mirror")
    plane = next(item for item in system.elements if item.kind == "plane_mirror")
    goal = next(item for item in system.elements if item.kind == "target")
    system.links = [
        ObjectLink(horn.uid, ellipse.uid),
        ObjectLink(ellipse.uid, plane.uid),
        ObjectLink(plane.uid, goal.uid),
    ]
    return system
