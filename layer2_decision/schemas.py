"""Validated data contracts for Layer 2 edit decisions."""
import math

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class EditSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self):
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("segment timestamps must be finite")
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")
        return self


class EditPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep_segments: list[EditSegment]
    remove_segments: list[EditSegment]
    flag_zoom: list[EditSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timeline(self, info: ValidationInfo):
        duration = (info.context or {}).get("duration")
        for field_name in ("keep_segments", "remove_segments", "flag_zoom"):
            segments = getattr(self, field_name)
            segments.sort(key=lambda segment: (segment.start, segment.end))
            for previous, current in zip(segments, segments[1:]):
                if current.start < previous.end:
                    raise ValueError(f"{field_name} contains overlapping segments")
            if duration is not None:
                for segment in segments:
                    if segment.end > duration + 0.05:
                        raise ValueError(
                            f"{field_name} segment ends after video duration ({duration:.3f}s)"
                        )

        for keep in self.keep_segments:
            for remove in self.remove_segments:
                if keep.start < remove.end and remove.start < keep.end:
                    raise ValueError("keep and remove segments overlap")

        if not self.keep_segments:
            raise ValueError("edit plan must contain at least one keep segment")
        return self


def validate_edit_plan(data: dict, duration: float | None = None) -> dict:
    """Validate and normalize an edit-plan dictionary."""
    plan = EditPlan.model_validate(data, context={"duration": duration})
    return plan.model_dump(mode="json")
