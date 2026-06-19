from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timedelta
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMPLOYEE_RECORDS = ROOT / "logs" / "employee_records.json"
DEFAULT_LIVE_STATS = ROOT / "logs" / "live_stats.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "reports"
MIN_DISPLAY_SECONDS = 1.0


def timestamped_report_path(output: Path | str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"employee_activity_report_{timestamp}.pdf"
    if output is None:
        return DEFAULT_OUTPUT / default_name
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() != ".pdf":
        return path / default_name
    return path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def employee_rows(records_payload: dict) -> list[dict]:
    rows = records_payload.get("employees") or records_payload.get("stats") or []
    return sorted(rows, key=lambda item: (str(item.get("current_camera", "")), str(item.get("name", ""))))


def seconds_text(value: float | int | str | None) -> str:
    seconds = max(0.0, float(value or 0.0))
    if seconds < MIN_DISPLAY_SECONDS:
        return "<1 sec"
    if seconds < 60:
        return f"{seconds:.0f} sec"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.2f} hr"


def meaningful_time_items(mapping: dict) -> list[tuple[str, float]]:
    items = [(str(k), float(v or 0.0)) for k, v in (mapping or {}).items()]
    items = [(key, value) for key, value in items if value >= MIN_DISPLAY_SECONDS]
    return sorted(items, key=lambda item: item[1], reverse=True)


class PdfWriter:
    def __init__(self) -> None:
        self.doc = fitz.open()
        self.page = None
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=595, height=842)
        self.y = 42

    def ensure_space(self, needed: float) -> None:
        if self.y + needed > 800:
            self.new_page()

    def text(self, text: str, x: float = 42, size: float = 10, bold: bool = False, color=(0, 0, 0)) -> None:
        self.ensure_space(size + 8)
        font = "hebo" if bold else "helv"
        self.page.insert_text((x, self.y), str(text), fontsize=size, fontname=font, color=color)
        self.y += size + 5

    def box(self, title: str, rows: list[tuple[str, str]], width: float = 510) -> None:
        row_h = 19
        height = 30 + max(1, len(rows)) * row_h
        self.ensure_space(height + 10)
        rect = fitz.Rect(42, self.y, 42 + width, self.y + height)
        self.page.draw_rect(rect, color=(0.82, 0.86, 0.8), fill=(0.98, 0.99, 0.97), width=0.8)
        self.page.insert_text((54, self.y + 19), title, fontsize=11, fontname="hebo", color=(0.16, 0.27, 0.2))
        y = self.y + 39
        if not rows:
            rows = [("No data", "-")]
        for label, value in rows:
            self.page.insert_text((54, y), str(label), fontsize=9.5, fontname="helv", color=(0.15, 0.15, 0.15))
            value_rect = fitz.Rect(225, y - 12, 540, y + 8)
            self.page.insert_textbox(value_rect, str(value), fontsize=9.5, fontname="hebo", align=fitz.TEXT_ALIGN_RIGHT)
            y += row_h
        self.y += height + 10

    def paragraph(self, title: str, text: str) -> None:
        self.ensure_space(80)
        self.text(title, size=11, bold=True)
        rect = fitz.Rect(54, self.y - 4, 545, self.y + 72)
        used = self.page.insert_textbox(rect, str(text), fontsize=9.5, fontname="helv", color=(0.1, 0.1, 0.1))
        self.y += max(26, 76 - max(0, used))

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(output_path)
        self.doc.close()


def summary_rows(employees: list[dict]) -> list[tuple[str, str]]:
    rows = []
    for employee in employees:
        camera_items = meaningful_time_items(employee.get("camera_times_sec") or {})
        primary_camera = camera_report_label(camera_items[0][0]) if camera_items else "-"
        active_time = seconds_text(visible_seconds(employee))
        working_time = sum(
            seconds
            for status, seconds in meaningful_time_items(employee.get("status_times_sec") or {})
            if "WORK" in status.upper()
        )
        idle_time = sum(
            seconds
            for status, seconds in meaningful_time_items(employee.get("status_times_sec") or {})
            if "IDLE" in status.upper() or "REST" in status.upper()
        )
        status_bits = [f"active {active_time}"]
        if working_time >= MIN_DISPLAY_SECONDS:
            status_bits.append(f"work {seconds_text(working_time)}")
        if idle_time >= MIN_DISPLAY_SECONDS:
            status_bits.append(f"idle {seconds_text(idle_time)}")
        rows.append(
            (
                f"{employee.get('id', '-')}: {employee.get('name', '-')}",
                f"{' | '.join(status_bits)} | {primary_camera} | {float(employee.get('total_distance_ft') or 0):.1f} ft",
            )
        )
    return rows


def camera_report_label(camera_id: str) -> str:
    labels = {
        "palletline": "Pallet Line",
        "sheetline": "Sheet Line",
        "main_production": "Main Production",
        "production2": "GC Production 2",
        "gc_production2": "GC Production 2",
        "production_outside": "Production Outside / Transition",
        "coolingroom2": "Cooling Room 2",
        "cooling_room2": "Cooling Room 2",
        "mixer1": "Mixer 1",
        "mixer2": "Mixer 2",
        "mixer3": "Mixer 3",
        "mixer4": "Mixer 4",
    }
    return labels.get(str(camera_id), str(camera_id).replace("_", " ").title())


def camera_activity_note(camera_id: str, rank: int) -> str:
    camera_id = str(camera_id)
    if camera_id == "production_outside":
        return "transition / walking between production areas"
    if "pallet" in camera_id:
        return "primary duty / pallet-line work area" if rank == 0 else "pallet-line support time"
    if "sheet" in camera_id:
        return "primary duty / sheet-line work area" if rank == 0 else "sheet-line support time"
    if "mixer" in camera_id:
        return "primary duty / mixer work area" if rank == 0 else "mixer support time"
    if "cooling" in camera_id:
        return "cooling-room task / room visit"
    if "production2" in camera_id or "gc_production2" in camera_id:
        return "GC Production 2 task area"
    if "main" in camera_id:
        return "main production task area"
    return "camera/zone presence time"


def visible_seconds(employee: dict) -> float:
    camera_total = sum(float(v or 0.0) for v in (employee.get("camera_times_sec") or {}).values())
    status_total = sum(float(v or 0.0) for v in (employee.get("status_times_sec") or {}).values())
    return max(camera_total, status_total)


def scheduled_break_windows(start_ts: float, end_ts: float) -> list[tuple[str, float, float]]:
    if end_ts <= start_ts:
        return []
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    windows = []
    day = start_dt.date()
    while day <= end_dt.date():
        base = datetime.combine(day, time.min)
        is_friday = base.weekday() == 4
        candidates = [
            ("Tea Break", base + timedelta(hours=9), base + timedelta(hours=9, minutes=15)),
            ("Lunch Break", base + timedelta(hours=13), base + timedelta(hours=14 if is_friday else 13, minutes=0 if is_friday else 30)),
            ("Tea Break", base + timedelta(hours=15), base + timedelta(hours=15, minutes=15)),
        ]
        for label, begin, finish in candidates:
            overlap_start = max(start_ts, begin.timestamp())
            overlap_end = min(end_ts, finish.timestamp())
            if overlap_end > overlap_start:
                windows.append((label, overlap_start, overlap_end))
        day += timedelta(days=1)
    return windows


def scheduled_break_seconds(employee: dict) -> dict[str, float]:
    first_seen = employee.get("first_seen_ts")
    last_seen = employee.get("last_seen_ts")
    if first_seen is None or last_seen is None:
        return {}
    totals: dict[str, float] = {}
    for label, begin, end in scheduled_break_windows(float(first_seen), float(last_seen)):
        totals[label] = totals.get(label, 0.0) + max(0.0, end - begin)
    return totals


def untracked_seconds(employee: dict) -> float:
    first_seen = employee.get("first_seen_ts")
    last_seen = employee.get("last_seen_ts")
    if first_seen is None or last_seen is None:
        return 0.0
    elapsed = max(0.0, float(last_seen) - float(first_seen))
    return max(0.0, elapsed - visible_seconds(employee) - sum(scheduled_break_seconds(employee).values()))


def client_breakdown_rows(employee: dict) -> list[tuple[str, str]]:
    rows = []
    camera_items = meaningful_time_items(employee.get("camera_times_sec") or {})
    for rank, (camera_id, seconds) in enumerate(camera_items):
        label = camera_report_label(camera_id)
        prefix = "Primary duty" if rank == 0 else "Secondary zone"
        rows.append((f"{prefix}: {label}", f"{seconds_text(seconds)} ({camera_activity_note(camera_id, rank)})"))

    status_items = dict(meaningful_time_items(employee.get("status_times_sec") or {}))
    walking = sum(seconds for status, seconds in status_items.items() if "WALK" in status.upper())
    working = sum(seconds for status, seconds in status_items.items() if "WORK" in status.upper())
    idle = sum(seconds for status, seconds in status_items.items() if "IDLE" in status.upper() or "REST" in status.upper())
    unknown = untracked_seconds(employee)
    breaks = scheduled_break_seconds(employee)

    if working >= MIN_DISPLAY_SECONDS:
        rows.append(("Work / active time", f"{seconds_text(working)} (hands/body/machine work or active task time)"))
    if walking >= MIN_DISPLAY_SECONDS:
        rows.append(("Transition / walking", f"{seconds_text(walking)} (moving between stations or zones)"))
    if idle >= MIN_DISPLAY_SECONDS:
        rows.append(("Rest / idle", f"{seconds_text(idle)} (visible but low movement)"))
    for label, seconds in meaningful_time_items(breaks):
        rows.append((label, f"{seconds_text(seconds)} (scheduled break time, not counted as tracking failure)"))
    if unknown >= MIN_DISPLAY_SECONDS:
        rows.append(("Untracked / not credited", f"{seconds_text(unknown)} (not confirmed by employee tracking during this period)"))

    if not rows:
        rows.append(("No confirmed employee activity", "No ArUco-identified employee time was available"))
    return rows


def build_pdf(records_payload: dict, live_payload: dict, output_path: Path) -> None:
    employees = employee_rows(records_payload)
    writer = PdfWriter()

    writer.text("Factory AI Employee Activity Report", size=18, bold=True)
    writer.box(
        "Report Snapshot",
        [
            ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Dashboard timestamp", records_payload.get("timestamp") or live_payload.get("timestamp") or "-"),
            ("Employees in report", str(len(employees))),
            ("Detected people count", str(live_payload.get("total_person_count", "-"))),
        ],
    )
    writer.box("Employee Summary", summary_rows(employees))

    for employee in employees:
        writer.new_page()
        writer.text(f"Employee {employee.get('id', '-')}: {employee.get('name', '-')}", size=16, bold=True)
        writer.box(
            "Current State",
            [
                ("Department", employee.get("dept", "-")),
                ("Current camera", employee.get("current_camera", "-")),
                ("Current zone", employee.get("zone", employee.get("zone_id", "-"))),
                ("Status", employee.get("status", "-")),
                ("Total distance", f"{float(employee.get('total_distance_ft') or 0):.2f} ft"),
                ("Visible tracked time", seconds_text(visible_seconds(employee))),
                ("Scheduled break time", seconds_text(sum(scheduled_break_seconds(employee).values()))),
                ("Untracked / not credited", seconds_text(untracked_seconds(employee))),
                ("Last seen", seconds_text(employee.get("last_seen_age"))),
                ("Person confidence", f"{float(employee.get('person_conf') or 0):.3f}"),
            ],
        )
        writer.box("Client Activity Breakdown", client_breakdown_rows(employee))
        writer.box(
            "Time By Camera",
            [(camera_id, seconds_text(seconds)) for camera_id, seconds in meaningful_time_items(employee.get("camera_times_sec") or {})],
        )
        writer.box(
            "Time By Activity",
            [(status, seconds_text(seconds)) for status, seconds in meaningful_time_items(employee.get("status_times_sec") or {})],
        )

    writer.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dashboard employee activity data to a PDF report")
    parser.add_argument("--employee-records", default=str(DEFAULT_EMPLOYEE_RECORDS))
    parser.add_argument("--live-stats", default=str(DEFAULT_LIVE_STATS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PDF path or folder. Folder output gets a timestamped filename.")
    args = parser.parse_args()

    records_payload = load_json(Path(args.employee_records))
    live_payload = load_json(Path(args.live_stats))
    if not employee_rows(records_payload):
        raise SystemExit(f"No employees found in {args.employee_records}. Run replay first and wait for detections.")

    output_path = timestamped_report_path(args.output)
    build_pdf(records_payload, live_payload, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
