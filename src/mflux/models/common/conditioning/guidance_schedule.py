class GuidanceSchedule:
    @staticmethod
    def parse_schedule(spec: str) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part or "-" not in part:
                raise ValueError(f"invalid guidance schedule segment {part!r}, expected 'start-end:value'")
            segment, value_str = part.rsplit(":", 1)
            start_str, end_str = segment.split("-", 1)
            start, end, value = float(start_str), float(end_str), float(value_str)
            if end < start:
                start, end = end, start
            points.append((start, end, value))
        if not points:
            raise ValueError(f"empty guidance schedule: {spec!r}")
        points.sort(key=lambda p: p[0])
        return points

    @staticmethod
    def schedule_to_per_step(points: list[tuple[float, float, float]], num_steps: int) -> list[float]:
        return [GuidanceSchedule._value_at(points, (i + 0.5) / num_steps) for i in range(num_steps)]

    @staticmethod
    def _value_at(points: list[tuple[float, float, float]], progress: float) -> float:
        for start, end, value in points:
            if start <= progress <= end:
                return value
        if progress < points[0][0]:
            return points[0][2]
        return points[-1][2]
