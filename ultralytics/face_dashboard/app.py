"""
Person & face recognition dashboard with entry counting.

Run: streamlit run face_dashboard/app.py
"""

from __future__ import annotations

import atexit
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_DIR = ROOT / "data" / "known_faces"
ENTRY_DIR = ROOT / "data" / "entries"
UNKNOWN_DIR = ROOT / "data" / "unknown_faces"
ACTIVITY_DIR = ROOT / "data" / "activity"
SAFETY_DIR = ROOT / "data" / "safety"

# Module-level camera ref so atexit can release hardware without Streamlit session.
_active_cap: cv2.VideoCapture | None = None
_shutting_down = False


def _track_cap(cap: cv2.VideoCapture | None) -> None:
    global _active_cap
    _active_cap = cap


def _release_camera_hardware() -> None:
    global _active_cap
    cap = _active_cap
    _active_cap = None
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def _on_process_exit() -> None:
    global _shutting_down
    _shutting_down = True
    _release_camera_hardware()


atexit.register(_on_process_exit)

if TYPE_CHECKING:
    from face_dashboard.activity_log import ActivityLogger
    from face_dashboard.entry_tracker import EntryTracker
    from face_dashboard.face_pipeline import FacePipeline
    from face_dashboard.face_store import FaceStore
    from face_dashboard.safety_monitor import SafetyMonitor


def _get_activity_logger() -> ActivityLogger:
    from face_dashboard.activity_log import ActivityLogger

    key = "activity_logger_v1"
    if key not in st.session_state:
        st.session_state[key] = ActivityLogger(ACTIVITY_DIR)
    return st.session_state[key]


def _get_safety_monitor() -> SafetyMonitor:
    from face_dashboard.safety_monitor import SafetyMonitor

    key = "safety_monitor_v1"
    if key not in st.session_state:
        dwell = st.session_state.get("unattended_dwell_sec", 30.0)
        st.session_state[key] = SafetyMonitor(SAFETY_DIR, dwell_sec=dwell)
    mon = st.session_state[key]
    mon.set_dwell_sec(st.session_state.get("unattended_dwell_sec", 30.0))
    mon.enable_unattended = st.session_state.get("enable_unattended", True)
    mon.enable_zones = st.session_state.get("enable_zones", True)
    mon.enable_fire_smoke = st.session_state.get("enable_fire_smoke", True)
    mon.enable_fight = st.session_state.get("enable_fight", True)
    mon.hazard_every = st.session_state.get("hazard_every", 6)
    return mon


def _get_unknown_store():
    from face_dashboard.unknown_face_store import UnknownFaceStore

    key = "unknown_store_v1"
    if key not in st.session_state:
        st.session_state[key] = UnknownFaceStore(UNKNOWN_DIR)
    return st.session_state[key]


def _get_entry_tracker() -> EntryTracker:
    from face_dashboard.entry_tracker import EntryTracker

    key = "entry_tracker_v4"
    st.session_state.pop("entry_tracker", None)
    st.session_state.pop("entry_tracker_v2", None)
    st.session_state.pop("entry_tracker_v3", None)
    if key not in st.session_state:
        st.session_state[key] = EntryTracker(
            ENTRY_DIR, unknown_store=_get_unknown_store()
        )
    return st.session_state[key]


def _entry_totals(tracker: EntryTracker) -> tuple[int, int, int]:
    visits = getattr(tracker, "visits", getattr(tracker, "entries", []))
    total = len(visits)
    identified = sum(1 for v in visits if v.entry_type == "identified")
    unknown = sum(1 for v in visits if v.entry_type == "unknown")
    return total, identified, unknown


def _reset_frame_cache(pipeline: object) -> None:
    if hasattr(pipeline, "reset_cache"):
        pipeline.reset_cache()
        return
    pipeline._frame_idx = 0
    pipeline._last_person_count = 0
    pipeline._last_person_boxes = None
    if hasattr(pipeline, "_last_objects"):
        pipeline._last_objects = []


def _release_camera() -> None:
    st.session_state.pop("cap", None)
    _release_camera_hardware()


def _resize_for_register(frame):
    h, w = frame.shape[:2]
    max_width = 640
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    work = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return work, 1.0 / scale


@st.cache_resource
def get_store() -> FaceStore:
    from face_dashboard.face_store import FaceStore

    return FaceStore(DB_DIR)


@st.cache_resource
def get_pipeline(_v: int = 10) -> FacePipeline:
    from face_dashboard.face_pipeline import FacePipeline

    return FacePipeline(get_store())


def page_header() -> None:
    st.set_page_config(page_title="Person & Face Dashboard", layout="wide", page_icon="👤")
    st.title("Person & Face Recognition Dashboard")
    st.caption("Campus safety · faces · bags · fire · smoke · fighting · zones · incidents")


def _read_process_config() -> SimpleNamespace:
    return SimpleNamespace(
        conf=st.session_state.get("conf", 0.35),
        max_width=st.session_state.get("max_width", 480),
        display_width=st.session_state.get("display_width", 640),
        imgsz=st.session_state.get("imgsz", 320),
        yolo_every=st.session_state.get("yolo_every", 99),
        faces_only=st.session_state.get("faces_only", True),
        jpeg_quality=st.session_state.get("jpeg_quality", 60),
        emotion_every=st.session_state.get("emotion_every", 3),
        behavior_every=st.session_state.get("behavior_every", 5),
        enable_emotion=st.session_state.get("enable_emotion", True),
        enable_behavior=st.session_state.get("enable_behavior", True),
        enable_objects=st.session_state.get("enable_objects", True),
        object_every=st.session_state.get("object_every", 5),
        enable_safety=st.session_state.get("enable_safety", True),
        unattended_dwell_sec=st.session_state.get("unattended_dwell_sec", 30.0),
        enable_fire_smoke=st.session_state.get("enable_fire_smoke", True),
        enable_fight=st.session_state.get("enable_fight", True),
        hazard_every=st.session_state.get("hazard_every", 6),
    )


def _metric_cards(
    persons: int,
    identified: int,
    unknown: int,
    objects: int,
    fps: str,
    entries_all: int,
    entries_id: int,
    entries_unk: int,
    entries_new: int,
) -> None:
    st.markdown("### In Camera Now")
    r1 = st.columns(5)
    labels_r1 = [
        ("Persons in frame", persons),
        ("Identified", identified),
        ("Unknown", unknown),
        ("Objects", objects),
        ("FPS", fps),
    ]
    for col, (label, val) in zip(r1, labels_r1):
        with col:
            with st.container(border=True):
                st.metric(label, val)

    st.markdown("### Entry Counts")
    r2 = st.columns(4)
    labels_r2 = [
        ("Total entries", entries_all),
        ("Identified entries", entries_id),
        ("Unknown entries", entries_unk),
        ("New entries", entries_new),
    ]
    for col, (label, val) in zip(r2, labels_r2):
        with col:
            with st.container(border=True):
                st.metric(label, val)


def _object_table(objects: list[Any]) -> None:
    if not objects:
        return
    st.markdown("### Objects in frame")
    rows = [
        {
            "Object": obj.label,
            "Confidence": f"{obj.confidence:.0%}",
            "BBox": f"{obj.bbox[0]},{obj.bbox[1]} → {obj.bbox[2]},{obj.bbox[3]}",
        }
        for obj in objects
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def _safety_alert_banner(snap: Any) -> None:
    if not snap or not snap.alerts:
        return
    st.markdown("### Safety alerts")
    for alert in snap.alerts[-5:]:
        if alert.severity == "high":
            st.error(alert.message)
        else:
            st.warning(alert.message)


def _incident_tables(monitor: SafetyMonitor) -> None:
    st.markdown("### Incident log")
    rows = monitor.incidents.recent_rows(15)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("No incidents logged yet.")

    counts = monitor.incidents.count_by_type()
    if counts:
        st.caption(
            "Totals: "
            + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in counts.items())
        )


def _activity_tables(logger: ActivityLogger) -> None:
    st.markdown("### Registered users — emotion & behavior")
    live = logger.summary_by_person()
    if live:
        st.dataframe(live, width="stretch", hide_index=True)
    else:
        st.caption("No activity logged yet for registered users.")

    st.markdown("**Recent emotion / behavior events**")
    recent = logger.recent_rows(12)
    if recent:
        st.dataframe(recent, width="stretch", hide_index=True)
    else:
        st.caption("Events appear when registered users are on camera.")


def _live_activity_cards(activity_snap: Any) -> None:
    if not activity_snap.live:
        return
    st.markdown("### Live — registered users")
    cols = st.columns(min(len(activity_snap.live), 4))
    for col, item in zip(cols, activity_snap.live):
        with col:
            with st.container(border=True):
                st.markdown(f"**{item.name}**")
                st.caption(f"Emotion: **{item.emotion}** ({item.emotion_conf:.0%})")
                st.caption(f"Behavior: **{item.behavior}** ({item.behavior_conf:.0%})")


def _entry_tables(tracker: EntryTracker) -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Visits per person**")
        rows = tracker.summary_rows()
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.caption("No visits yet.")
        in_now = tracker.currently_in_rows()
        if in_now:
            st.markdown("**Currently in camera**")
            st.dataframe(in_now, width="stretch", hide_index=True)
    with col_b:
        st.markdown("**In / out log**")
        rows = tracker.visit_rows(20)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.caption("Waiting for visits...")


def _render_idle_dashboard() -> None:
    tracker = _get_entry_tracker()
    total, identified, unknown = _entry_totals(tracker)

    st.markdown("### Live Camera")
    st.info("Click **Start camera** in the sidebar.")

    _metric_cards(0, 0, 0, 0, "—", total, identified, unknown, 0)
    _entry_tables(tracker)
    _activity_tables(_get_activity_logger())


@st.fragment(run_every=timedelta(seconds=1))
def _render_live_dashboard() -> None:
    """All live UI lives inside this fragment — avoids Streamlit delta path errors."""
    if _shutting_down or not st.session_state.get("monitoring"):
        return

    from face_dashboard.face_pipeline import read_camera_frame

    pipeline = get_pipeline()
    tracker = _get_entry_tracker()
    activity = _get_activity_logger()
    safety = _get_safety_monitor()
    cap = st.session_state.get("cap")

    st.markdown("### Live Camera")

    if cap is None or not cap.isOpened():
        st.warning("Camera lost. Click **Start camera** again.")
        return

    ok, frame = read_camera_frame(cap)
    if not ok or frame is None:
        st.warning("Bad camera frame — try **Stop** then **Start**, or change camera index.")
        return

    cfg = _read_process_config()
    result = pipeline.process_frame(frame, cfg)
    safety_snap = None
    if st.session_state.get("enable_safety", True):
        safety_snap = safety.update(result, frame, cfg=cfg)
    entry_stats = tracker.update(result.faces)
    activity_snap = activity.update(result.faces)

    rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
    st.image(rgb, width="stretch")

    _metric_cards(
        result.person_count,
        result.known_count,
        result.unknown_count,
        result.object_count,
        f"{result.fps:.1f}",
        entry_stats.total_entries,
        entry_stats.identified_entries,
        entry_stats.unknown_entries,
        len(entry_stats.new_entries_this_frame),
    )

    _object_table(result.objects)
    _safety_alert_banner(safety_snap)

    _live_activity_cards(activity_snap)

    st.session_state["tick"] = st.session_state.get("tick", 0) + 1
    if st.session_state["tick"] % 3 == 0:
        _entry_tables(tracker)
        _activity_tables(activity)
        if st.session_state.get("enable_safety", True):
            _incident_tables(safety)


def tab_live_monitor(pipeline: Any) -> None:
    from face_dashboard.face_pipeline import open_camera, warm_up_camera

    col_cfg, col_main = st.columns([1, 3])

    with col_cfg:
        st.markdown("**Controls**")
        camera = st.number_input("Camera index", 0, 5, 0, step=1)
        st.session_state["conf"] = st.slider("Confidence", 0.1, 1.0, 0.35, 0.05)

        preset = st.selectbox("Speed", ["Fastest", "Fast", "Balanced"], index=0)
        if preset == "Fast":
            st.session_state.update(
                max_width=480, display_width=720, imgsz=320, yolo_every=5,
                faces_only=False, jpeg_quality=65, enable_objects=True, object_every=5,
            )
        elif preset == "Balanced":
            st.session_state.update(
                max_width=640, display_width=960, imgsz=416, yolo_every=3,
                faces_only=False, jpeg_quality=72, enable_objects=True, object_every=3,
            )
        else:
            st.session_state.update(
                max_width=480, display_width=640, imgsz=320, yolo_every=99,
                faces_only=True, jpeg_quality=60, enable_objects=True, object_every=6,
            )

        stop = st.button("Stop camera", use_container_width=True, key="btn_stop_cam")
        start = st.button("Start camera", type="primary", use_container_width=True, key="btn_start_cam")

        if st.session_state.get("monitoring"):
            st.success("Camera ON")
        else:
            st.info("Camera OFF")

        if st.button("Reset entries", use_container_width=True, key="btn_reset_entries"):
            _get_entry_tracker().reset_session()
            _get_activity_logger().reset()
            st.rerun()

        if st.button("Reset incidents", use_container_width=True, key="btn_reset_incidents"):
            _get_safety_monitor().reset_session()
            st.rerun()

        st.markdown("**Safety**")
        st.session_state["enable_safety"] = st.checkbox(
            "Safety monitoring", st.session_state.get("enable_safety", True)
        )
        st.session_state["enable_unattended"] = st.checkbox(
            "Unattended bag alert", st.session_state.get("enable_unattended", True)
        )
        st.session_state["enable_zones"] = st.checkbox(
            "Zone alerts (restricted + crowd)", st.session_state.get("enable_zones", True)
        )
        st.session_state["enable_fire_smoke"] = st.checkbox(
            "Fire & smoke detection", st.session_state.get("enable_fire_smoke", True)
        )
        st.session_state["enable_fight"] = st.checkbox(
            "Fighting detection", st.session_state.get("enable_fight", True)
        )
        st.session_state["hazard_every"] = st.slider(
            "Hazard scan interval (frames)", 3, 15, int(st.session_state.get("hazard_every", 6)), 1
        )
        st.session_state["unattended_dwell_sec"] = st.slider(
            "Unattended threshold (sec)", 10, 120, int(st.session_state.get("unattended_dwell_sec", 30)), 5
        )

        st.session_state["enable_emotion"] = st.checkbox(
            "Emotion detection (registered only)", st.session_state.get("enable_emotion", True)
        )
        st.session_state["enable_behavior"] = st.checkbox(
            "Behavior detection (registered only)", st.session_state.get("enable_behavior", True)
        )
        st.session_state["enable_objects"] = st.checkbox(
            "Object detection (YOLO)", st.session_state.get("enable_objects", True)
        )

    with col_main:
        if stop:
            st.session_state["monitoring"] = False
            _release_camera()
            _reset_frame_cache(pipeline)
            st.rerun()

        if start:
            _release_camera()
            cap_w = min(st.session_state.get("max_width", 480), 640)
            cap = open_camera(int(camera), width=cap_w, height=int(cap_w * 0.75))
            if not cap.isOpened():
                st.error(f"Cannot open camera {camera}. Close Teams/Zoom.")
            else:
                st.session_state["cap"] = cap
                _track_cap(cap)
                st.session_state["monitoring"] = True
                st.session_state["tick"] = 0
                _reset_frame_cache(pipeline)
                warm_up_camera(cap, n=25)
                st.rerun()

        if st.session_state.get("monitoring"):
            _render_live_dashboard()
        else:
            _render_idle_dashboard()


def tab_register(pipeline: Any, store: Any) -> None:
    from face_dashboard.face_pipeline import open_camera

    st.subheader("Register Face")
    camera = st.number_input("Camera index", 0, 5, 0, step=1, key="reg_camera")
    name = st.text_input("Person name", placeholder="e.g. Rahul Sharma")
    face_index = st.number_input("Face index", 0, 10, 0, step=1)

    if st.button("Capture from webcam", type="primary"):
        cap = open_camera(int(camera), width=640, height=480)
        if cap.isOpened():
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if ok:
                st.session_state["register_frame"] = frame
                st.success("Captured!")
            else:
                st.error("Capture failed.")
        else:
            st.error("Cannot open camera.")

    upload = st.camera_input("Or upload photo")
    if upload is not None:
        import numpy as np
        from PIL import Image

        st.session_state["register_frame"] = cv2.cvtColor(np.array(Image.open(upload)), cv2.COLOR_RGB2BGR)

    frame = st.session_state.get("register_frame")
    if frame is None:
        st.info("Capture or upload a photo first.")
        return

    preview = frame.copy()
    work, inv = _resize_for_register(preview)
    faces = pipeline.detect_faces(work, inv)
    for i, face in enumerate(faces):
        x, y, fw, fh = face.bbox
        cv2.rectangle(preview, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    st.image(preview, channels="BGR", width="stretch", caption=f"{len(faces)} face(s)")

    if name.strip() and st.button("Save person", type="primary"):
        if not faces:
            st.error("No face detected.")
        elif pipeline.register_face(frame, name.strip(), int(face_index)):
            get_pipeline.clear()
            st.success(f"Saved **{name.strip()}**")
            del st.session_state["register_frame"]
            st.rerun()
        else:
            st.error("Save failed.")


def tab_safety() -> None:
    monitor = _get_safety_monitor()
    st.subheader("Campus Safety Zones")
    st.caption(
        "Orange box = **Restricted Lab** (unknown faces trigger alert). "
        "Yellow box = **Main Hall** crowd limit. "
        "Red box = unattended bag. Orange = bag tracked. Red labels = fire / fighting. Gray = smoke."
    )

    crowd_max = st.slider(
        "Main Hall max persons",
        2, 20, int(next((z.max_persons for z in monitor.zones.zones if z.id == "main_hall"), 8)),
        key="crowd_max_slider",
    )
    for zone in monitor.zones.zones:
        if zone.id == "main_hall":
            zone.max_persons = crowd_max
            monitor.zones.save()

    restricted_on = st.checkbox(
        "Restricted Lab zone enabled",
        value=next((z.enabled for z in monitor.zones.zones if z.id == "restricted_lab"), True),
        key="restricted_zone_on",
    )
    for zone in monitor.zones.zones:
        if zone.id == "restricted_lab":
            zone.enabled = restricted_on
            monitor.zones.save()

    st.markdown("**Configured zones**")
    zone_rows = [
        {
            "Zone": z.name,
            "Type": z.zone_type,
            "Enabled": z.enabled,
            "Max persons": z.max_persons if z.zone_type == "crowd" else "—",
            "Allow unknown": z.allow_unknown,
            "Area": f"({z.x1:.0%},{z.y1:.0%})→({z.x2:.0%},{z.y2:.0%})",
        }
        for z in monitor.zones.zones
    ]
    st.dataframe(zone_rows, width="stretch", hide_index=True)

    _incident_tables(monitor)


def tab_manage(store: Any) -> None:
    st.subheader("Registered People")
    if not store.faces:
        st.info("No one registered yet.")
    else:
        for face in store.faces:
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                if face.image_bgr is not None:
                    st.image(face.image_bgr, channels="BGR", width=80)
            with c2:
                st.markdown(f"**{face.name}**")
                st.caption(f"ID: `{face.id}`")
            with c3:
                if st.button("Delete", key=f"del_{face.id}"):
                    store.remove(face.id)
                    get_store.clear()
                    get_pipeline.clear()
                    st.rerun()

    st.divider()
    st.subheader("Unknown Visitors")
    unknown_store = _get_unknown_store()
    unknowns = unknown_store.list_all()
    if not unknowns:
        st.info("No unknown visitors captured yet. They are saved automatically from the live camera.")
    else:
        for person in unknowns:
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                if person.image_path and Path(person.image_path).exists():
                    st.image(person.image_path, width=80)
            with c2:
                st.markdown(f"**Unknown visitor**")
                st.caption(f"ID: `{person.id}` · Visits: {person.visit_count}")
                st.caption(f"Last seen: {person.last_seen}")
            with c3:
                if st.button("Delete", key=f"del_unk_{person.id}"):
                    unknown_store.remove(person.id)
                    st.rerun()


def main() -> None:
    page_header()
    st.session_state.setdefault("monitoring", False)
    _get_entry_tracker()
    _get_activity_logger()

    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Register Face", "Manage People", "Safety"])
    with tab1:
        tab_live_monitor(get_pipeline())
    with tab2:
        tab_register(get_pipeline(), get_store())
    with tab3:
        tab_manage(get_store())
    with tab4:
        tab_safety()


if __name__ == "__main__":
    main()
