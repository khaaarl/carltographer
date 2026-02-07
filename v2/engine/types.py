"""Data types matching the carltographer JSON schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Transform:
    x: float = 0.0
    z: float = 0.0
    rotation_deg: float = 0.0

    @staticmethod
    def from_dict(d: dict | None) -> Transform:
        if not d:
            return Transform()
        return Transform(
            x=d.get("x_inches", 0.0),
            z=d.get("z_inches", 0.0),
            rotation_deg=d.get("rotation_deg", 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "x_inches": self.x,
            "y_inches": 0.0,
            "z_inches": self.z,
            "rotation_deg": self.rotation_deg,
        }


@dataclass
class Shape:
    width: float
    depth: float
    height: float
    offset: Transform | None = None

    @staticmethod
    def from_dict(d: dict) -> Shape:
        offset_d = d.get("offset")
        return Shape(
            width=d["width_inches"],
            depth=d["depth_inches"],
            height=d["height_inches"],
            offset=(Transform.from_dict(offset_d) if offset_d else None),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "shape_type": "rectangular_prism",
            "width_inches": self.width,
            "depth_inches": self.depth,
            "height_inches": self.height,
        }
        if self.offset:
            d["offset"] = self.offset.to_dict()
        return d


@dataclass
class TerrainObject:
    id: str
    shapes: list[Shape]
    name: str | None = None
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> TerrainObject:
        return TerrainObject(
            id=d["id"],
            shapes=[Shape.from_dict(s) for s in d["shapes"]],
            name=d.get("name"),
            tags=d.get("tags", []),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "shapes": [s.to_dict() for s in self.shapes],
        }
        if self.name:
            d["name"] = self.name
        if self.tags:
            d["tags"] = self.tags
        return d


@dataclass
class FeatureComponent:
    object_id: str
    transform: Transform | None = None

    @staticmethod
    def from_dict(d: dict) -> FeatureComponent:
        t = d.get("transform")
        return FeatureComponent(
            object_id=d["object_id"],
            transform=Transform.from_dict(t) if t else None,
        )

    def to_dict(self) -> dict:
        d: dict = {"object_id": self.object_id}
        if self.transform:
            d["transform"] = self.transform.to_dict()
        return d


@dataclass
class TerrainFeature:
    id: str
    feature_type: str
    components: list[FeatureComponent]

    @staticmethod
    def from_dict(d: dict) -> TerrainFeature:
        return TerrainFeature(
            id=d["id"],
            feature_type=d["feature_type"],
            components=[
                FeatureComponent.from_dict(c) for c in d["components"]
            ],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "feature_type": self.feature_type,
            "components": [c.to_dict() for c in self.components],
        }


@dataclass
class PlacedFeature:
    feature: TerrainFeature
    transform: Transform

    @staticmethod
    def from_dict(d: dict) -> PlacedFeature:
        return PlacedFeature(
            feature=TerrainFeature.from_dict(d["feature"]),
            transform=Transform.from_dict(d.get("transform")),
        )

    def to_dict(self) -> dict:
        return {
            "feature": self.feature.to_dict(),
            "transform": self.transform.to_dict(),
        }


@dataclass
class TerrainLayout:
    table_width: float
    table_depth: float
    placed_features: list[PlacedFeature] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> TerrainLayout:
        return TerrainLayout(
            table_width=d["table_width_inches"],
            table_depth=d["table_depth_inches"],
            placed_features=[
                PlacedFeature.from_dict(p)
                for p in d.get("placed_features", [])
            ],
        )

    def to_dict(self) -> dict:
        return {
            "table_width_inches": self.table_width,
            "table_depth_inches": self.table_depth,
            "placed_features": [p.to_dict() for p in self.placed_features],
        }


@dataclass
class CatalogObject:
    item: TerrainObject
    quantity: int | None = None

    @staticmethod
    def from_dict(d: dict) -> CatalogObject:
        return CatalogObject(
            item=TerrainObject.from_dict(d["item"]),
            quantity=d.get("quantity"),
        )


@dataclass
class CatalogFeature:
    item: TerrainFeature
    quantity: int | None = None

    @staticmethod
    def from_dict(d: dict) -> CatalogFeature:
        return CatalogFeature(
            item=TerrainFeature.from_dict(d["item"]),
            quantity=d.get("quantity"),
        )


@dataclass
class TerrainCatalog:
    objects: list[CatalogObject] = field(default_factory=list)
    features: list[CatalogFeature] = field(default_factory=list)
    name: str | None = None

    @staticmethod
    def from_dict(d: dict) -> TerrainCatalog:
        return TerrainCatalog(
            objects=[CatalogObject.from_dict(o) for o in d.get("objects", [])],
            features=[
                CatalogFeature.from_dict(f) for f in d.get("features", [])
            ],
            name=d.get("name"),
        )


@dataclass
class FeatureCountPreference:
    feature_type: str
    min: int = 0
    max: int | None = None

    @staticmethod
    def from_dict(d: dict) -> FeatureCountPreference:
        return FeatureCountPreference(
            feature_type=d["feature_type"],
            min=d.get("min", 0),
            max=d.get("max"),
        )


@dataclass
class EngineParams:
    seed: int
    table_width: float
    table_depth: float
    catalog: TerrainCatalog
    num_steps: int = 100
    initial_layout: TerrainLayout | None = None
    feature_count_preferences: list[FeatureCountPreference] = field(
        default_factory=list
    )

    @staticmethod
    def from_dict(d: dict) -> EngineParams:
        il = d.get("initial_layout")
        prefs = [
            FeatureCountPreference.from_dict(p)
            for p in d.get("feature_count_preferences", [])
        ]
        return EngineParams(
            seed=d["seed"],
            table_width=d["table_width_inches"],
            table_depth=d["table_depth_inches"],
            catalog=TerrainCatalog.from_dict(d["catalog"]),
            num_steps=d.get("num_steps", 100),
            initial_layout=(TerrainLayout.from_dict(il) if il else None),
            feature_count_preferences=prefs,
        )


@dataclass
class EngineResult:
    layout: TerrainLayout
    score: float = 0.0
    steps_completed: int = 0

    def to_dict(self) -> dict:
        return {
            "layout": self.layout.to_dict(),
            "score": self.score,
            "steps_completed": self.steps_completed,
        }
