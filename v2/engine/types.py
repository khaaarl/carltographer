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
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> TerrainFeature:
        return TerrainFeature(
            id=d["id"],
            feature_type=d["feature_type"],
            components=[
                FeatureComponent.from_dict(c) for c in d["components"]
            ],
            tags=d.get("tags", []),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "feature_type": self.feature_type,
            "components": [c.to_dict() for c in self.components],
        }
        if self.tags:
            d["tags"] = self.tags
        return d


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
    rotationally_symmetric: bool = False
    visibility: dict | None = None
    mission: Mission | None = None
    terrain_objects: list[TerrainObject] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> TerrainLayout:
        m = d.get("mission")
        return TerrainLayout(
            table_width=d["table_width_inches"],
            table_depth=d["table_depth_inches"],
            placed_features=[
                PlacedFeature.from_dict(p)
                for p in d.get("placed_features", [])
            ],
            rotationally_symmetric=d.get("rotationally_symmetric", False),
            visibility=d.get("visibility"),
            mission=Mission.from_dict(m) if m else None,
            terrain_objects=[
                TerrainObject.from_dict(o)
                for o in d.get("terrain_objects", [])
            ],
        )

    def to_dict(self) -> dict:
        d = {
            "table_width_inches": self.table_width,
            "table_depth_inches": self.table_depth,
            "placed_features": [p.to_dict() for p in self.placed_features],
            "rotationally_symmetric": self.rotationally_symmetric,
        }
        if self.terrain_objects:
            d["terrain_objects"] = [o.to_dict() for o in self.terrain_objects]
        if self.visibility is not None:
            d["visibility"] = self.visibility
        if self.mission is not None:
            d["mission"] = self.mission.to_dict()
        return d


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
class Point2D:
    x: float
    z: float

    @staticmethod
    def from_dict(d: dict) -> Point2D:
        return Point2D(x=d["x_inches"], z=d["z_inches"])

    def to_dict(self) -> dict:
        return {"x_inches": self.x, "z_inches": self.z}


@dataclass
class ObjectiveMarker:
    id: str
    position: Point2D
    range_inches: float = 3.0

    @staticmethod
    def from_dict(d: dict) -> ObjectiveMarker:
        return ObjectiveMarker(
            id=d["id"],
            position=Point2D.from_dict(d["position"]),
            range_inches=d.get("range_inches", 3.0),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "range_inches": self.range_inches,
        }


@dataclass
class DeploymentZone:
    id: str
    polygons: list[list[Point2D]]

    @staticmethod
    def from_dict(d: dict) -> DeploymentZone:
        polygons = []
        for poly in d.get("polygons", []):
            polygons.append([Point2D.from_dict(p) for p in poly])
        return DeploymentZone(id=d["id"], polygons=polygons)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "polygons": [
                [p.to_dict() for p in poly] for poly in self.polygons
            ],
        }


@dataclass
class Mission:
    name: str
    objectives: list[ObjectiveMarker]
    deployment_zones: list[DeploymentZone]
    rotationally_symmetric: bool = False

    @staticmethod
    def from_dict(d: dict) -> Mission:
        return Mission(
            name=d.get("name", ""),
            objectives=[
                ObjectiveMarker.from_dict(o) for o in d.get("objectives", [])
            ],
            deployment_zones=[
                DeploymentZone.from_dict(dz)
                for dz in d.get("deployment_zones", [])
            ],
            rotationally_symmetric=d.get("rotationally_symmetric", False),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "objectives": [o.to_dict() for o in self.objectives],
            "deployment_zones": [dz.to_dict() for dz in self.deployment_zones],
            "rotationally_symmetric": self.rotationally_symmetric,
        }


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
class ScoringTargets:
    overall_visibility_target: float | None = None
    overall_visibility_weight: float = 1.0
    dz_visibility_target: float | None = None
    dz_visibility_weight: float = 1.0
    dz_hidden_target: float | None = None
    dz_hidden_weight: float = 1.0
    objective_hidability_target: float | None = None
    objective_hidability_weight: float = 1.0

    @staticmethod
    def from_dict(d: dict) -> ScoringTargets:
        return ScoringTargets(
            overall_visibility_target=d.get("overall_visibility_target"),
            overall_visibility_weight=d.get("overall_visibility_weight", 1.0),
            dz_visibility_target=d.get("dz_visibility_target"),
            dz_visibility_weight=d.get("dz_visibility_weight", 1.0),
            dz_hidden_target=d.get("dz_hidden_target"),
            dz_hidden_weight=d.get("dz_hidden_weight", 1.0),
            objective_hidability_target=d.get("objective_hidability_target"),
            objective_hidability_weight=d.get(
                "objective_hidability_weight", 1.0
            ),
        )

    def to_dict(self) -> dict:
        d: dict = {}
        if self.overall_visibility_target is not None:
            d["overall_visibility_target"] = self.overall_visibility_target
            d["overall_visibility_weight"] = self.overall_visibility_weight
        if self.dz_visibility_target is not None:
            d["dz_visibility_target"] = self.dz_visibility_target
            d["dz_visibility_weight"] = self.dz_visibility_weight
        if self.dz_hidden_target is not None:
            d["dz_hidden_target"] = self.dz_hidden_target
            d["dz_hidden_weight"] = self.dz_hidden_weight
        if self.objective_hidability_target is not None:
            d["objective_hidability_target"] = self.objective_hidability_target
            d["objective_hidability_weight"] = self.objective_hidability_weight
        return d


@dataclass
class TuningParams:
    max_retries: int = 100
    retry_decay: float = 0.95
    min_move_range: float = 2.0
    max_extra_mutations: int = 3
    tile_size: float = 2.0
    delete_weight_last: float = 0.25
    rotate_on_move_prob: float = 0.5
    shortage_boost: float = 2.0
    excess_boost: float = 2.0
    penalty_factor: float = 0.1
    phase2_base: float = 1000.0
    temp_ladder_min_ratio: float = 0.01

    @staticmethod
    def from_dict(d: dict) -> TuningParams:
        return TuningParams(
            max_retries=d.get("max_retries", 100),
            retry_decay=d.get("retry_decay", 0.95),
            min_move_range=d.get("min_move_range", 2.0),
            max_extra_mutations=d.get("max_extra_mutations", 3),
            tile_size=d.get("tile_size", 2.0),
            delete_weight_last=d.get("delete_weight_last", 0.25),
            rotate_on_move_prob=d.get("rotate_on_move_prob", 0.5),
            shortage_boost=d.get("shortage_boost", 2.0),
            excess_boost=d.get("excess_boost", 2.0),
            penalty_factor=d.get("penalty_factor", 0.1),
            phase2_base=d.get("phase2_base", 1000.0),
            temp_ladder_min_ratio=d.get("temp_ladder_min_ratio", 0.01),
        )

    def to_dict(self) -> dict:
        return {
            "max_retries": self.max_retries,
            "retry_decay": self.retry_decay,
            "min_move_range": self.min_move_range,
            "max_extra_mutations": self.max_extra_mutations,
            "tile_size": self.tile_size,
            "delete_weight_last": self.delete_weight_last,
            "rotate_on_move_prob": self.rotate_on_move_prob,
            "shortage_boost": self.shortage_boost,
            "excess_boost": self.excess_boost,
            "penalty_factor": self.penalty_factor,
            "phase2_base": self.phase2_base,
            "temp_ladder_min_ratio": self.temp_ladder_min_ratio,
        }


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
    min_feature_gap_inches: float | None = None
    min_edge_gap_inches: float | None = None
    min_all_feature_gap_inches: float | None = None
    min_all_edge_gap_inches: float | None = None
    rotation_granularity_deg: float = 15.0
    rotationally_symmetric: bool = False
    mission: Mission | None = None
    skip_visibility: bool = False
    scoring_targets: ScoringTargets | None = None
    num_replicas: int | None = None
    swap_interval: int = 20
    max_temperature: float = 50.0
    tuning: TuningParams | None = None

    def get_tuning(self) -> TuningParams:
        return self.tuning if self.tuning is not None else TuningParams()

    @staticmethod
    def from_dict(d: dict) -> EngineParams:
        il = d.get("initial_layout")
        m = d.get("mission")
        st = d.get("scoring_targets")
        tu = d.get("tuning")
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
            min_feature_gap_inches=d.get("min_feature_gap_inches"),
            min_edge_gap_inches=d.get("min_edge_gap_inches"),
            min_all_feature_gap_inches=d.get("min_all_feature_gap_inches"),
            min_all_edge_gap_inches=d.get("min_all_edge_gap_inches"),
            rotation_granularity_deg=d.get("rotation_granularity_deg", 15.0),
            rotationally_symmetric=d.get("rotationally_symmetric", False),
            mission=Mission.from_dict(m) if m else None,
            skip_visibility=d.get("skip_visibility", False),
            scoring_targets=(ScoringTargets.from_dict(st) if st else None),
            num_replicas=d.get("num_replicas"),
            swap_interval=d.get("swap_interval", 20),
            max_temperature=d.get("max_temperature", 50.0),
            tuning=TuningParams.from_dict(tu) if tu else None,
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
