from __future__ import annotations
import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import folium
import requests
from PIL import Image
from io import BytesIO

INPUT_FILE = "FLIGHT66.BIN"
OUTPUT_DIR = None
MAKE_PLOTS = True
MAKE_VIDEO = True
VIDEO_FPS = 30
VIDEO_MAX_FRAMES = 300
ROCKET_NOSE_AXIS = "-X"
BODY_X_COLOR = "red"
BODY_Y_COLOR = "green"
BODY_Z_COLOR = "blue"
ROCKET_NOSE_COLOR = "purple"
TRAJECTORY_COLOR = "black"
CURRENT_POSITION_COLOR = "orange"
BODY_LINE_WIDTH = 3.0
NOSE_LINE_WIDTH = 4.0
TRAJECTORY_LINE_WIDTH = 2.0
BODY_AXIS_LENGTH_M = 2.0
NOSE_AXIS_LENGTH_M = 2.5
AXIS_VISUAL_FRACTION_OF_PLOT = 0.18
AXIS_TIP_MARKER_SIZE = 65
CENTER_MARKER_SIZE = 55
AXIS_LABEL_OFFSET_FRACTION = 0.14
SHOW_NEGATIVE_BODY_AXES = True
NEGATIVE_AXIS_ALPHA = 0.28
ROCKET_AXIS_ELEV = 22
ROCKET_AXIS_AZIM = -55
MAKE_SATELLITE_GPS_MAP = True
MAKE_SATELLITE_GPS_ANIMATION = True
SATELLITE_ANIMATION_FPS = 20
SATELLITE_ANIMATION_MAX_POINTS = 900
SATELLITE_ANIMATION_PLAYBACK_SPEED = 1.0
MAKE_SATELLITE_GPS_VIDEO = True
SATELLITE_VIDEO_FPS = 20
SATELLITE_VIDEO_MAX_FRAMES = 450
SATELLITE_VIDEO_MIN_ZOOM = 12
SATELLITE_VIDEO_MAX_ZOOM = 19
SATELLITE_VIDEO_MAX_TILES = 90
SATELLITE_VIDEO_DPI = 130
MAX_ORIENTATION_FRAMES = 150
MAKE_POSITION_ORIENTATION_VIDEO = True
MAKE_POSITION_ORIENTATION_FRAMES = False
MAX_POSITION_ORIENTATION_FRAMES = 150

SCRIPT_DIR = Path(__file__).resolve().parent
HEADER_STRUCT = struct.Struct("<4sHHHHI")

RECORD_STRUCT_V1 = struct.Struct(
    "<"
    "II"
    "hhhhhh"
    "hhh"
    "hIi"
    "BBiiiHII"
    "BBiiiHII"
    "B"
    "iiii"
    "hhhhhh"
    "H"
)

RECORD_STRUCT_V2 = struct.Struct(
    "<"
    "II"
    "hhhhhh"
    "hhh"
    "hIi"
    "BBiiiHII"
    "BBiiiHII"
    "B"
    "iiii"
    "hhhhhh"
    "BB"
    "iiiiiiiiiiii"
    "H"
)

CSV_COLUMNS = [
    "seq",
    "ms",
    "time_s",

    "icm_ax_g",
    "icm_ay_g",
    "icm_az_g",
    "icm_gx_dps",
    "icm_gy_dps",
    "icm_gz_dps",

    "adxl_ax_g",
    "adxl_ay_g",
    "adxl_az_g",

    "bmp_temp_c",
    "bmp_press_pa",
    "bmp_alt_m",

    "gps1_fix",
    "gps1_sats",
    "gps1_lat_deg",
    "gps1_lon_deg",
    "gps1_alt_m",
    "gps1_speed_knots",
    "gps1_time_utc",
    "gps1_date_utc",

    "gps2_fix",
    "gps2_sats",
    "gps2_lat_deg",
    "gps2_lon_deg",
    "gps2_alt_m",
    "gps2_speed_knots",
    "gps2_time_utc",
    "gps2_date_utc",

    "attitude_valid",
    "q0",
    "q1",
    "q2",
    "q3",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "gyro_bias_x_dps",
    "gyro_bias_y_dps",
    "gyro_bias_z_dps",

    "nav_valid",
    "nav_gps_source",
    "nav_x_m",
    "nav_y_m",
    "nav_z_m",
    "nav_vx_mps",
    "nav_vy_mps",
    "nav_vz_mps",
    "nav_ax_mps2",
    "nav_ay_mps2",
    "nav_az_mps2",
    "nav_lat_deg",
    "nav_lon_deg",
    "nav_alt_m",

    "crc16_recorded",
    "crc16_computed",
    "crc_ok",
]

@dataclass
class Header:
    magic: bytes
    version: int
    header_size: int
    record_size: int
    reserved: int
    start_ms: int

def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF

def parse_nmea_u32(value: int) -> str:
    if value == 0:
        return ""
    return str(value)

def read_header(f) -> Header:
    raw = f.read(HEADER_STRUCT.size)
    if len(raw) != HEADER_STRUCT.size:
        raise ValueError("File is too small to contain a valid header.")
    magic, version, header_size, record_size, reserved, start_ms = HEADER_STRUCT.unpack(raw)
    return Header(
        magic=magic,
        version=version,
        header_size=header_size,
        record_size=record_size,
        reserved=reserved,
        start_ms=start_ms,
    )

def empty_nav_fields() -> dict:
    return {
        "nav_valid": 0,
        "nav_gps_source": 0,
        "nav_x_m": 0.0,
        "nav_y_m": 0.0,
        "nav_z_m": 0.0,
        "nav_vx_mps": 0.0,
        "nav_vy_mps": 0.0,
        "nav_vz_mps": 0.0,
        "nav_ax_mps2": 0.0,
        "nav_ay_mps2": 0.0,
        "nav_az_mps2": 0.0,
        "nav_lat_deg": 0.0,
        "nav_lon_deg": 0.0,
        "nav_alt_m": 0.0,
    }

def decode_record(raw: bytes, record_size: int) -> dict:
    if record_size == RECORD_STRUCT_V1.size:
        values = RECORD_STRUCT_V1.unpack(raw)
        is_v2 = False
    elif record_size == RECORD_STRUCT_V2.size:
        values = RECORD_STRUCT_V2.unpack(raw)
        is_v2 = True
    else:
        raise ValueError(
            f"Unsupported record length: {record_size}. "
            f"V1 expects {RECORD_STRUCT_V1.size}, V2 expects {RECORD_STRUCT_V2.size}."
        )
    i = 0
    seq = values[i]; i += 1
    ms = values[i]; i += 1
    icm_ax_mg = values[i]; i += 1
    icm_ay_mg = values[i]; i += 1
    icm_az_mg = values[i]; i += 1
    icm_gx_cdegps = values[i]; i += 1
    icm_gy_cdegps = values[i]; i += 1
    icm_gz_cdegps = values[i]; i += 1
    adxl_ax_cg = values[i]; i += 1
    adxl_ay_cg = values[i]; i += 1
    adxl_az_cg = values[i]; i += 1
    bmp_temp_cC = values[i]; i += 1
    bmp_press_pa = values[i]; i += 1
    bmp_alt_cm = values[i]; i += 1
    gps1_fix = values[i]; i += 1
    gps1_sats = values[i]; i += 1
    gps1_lat_e7 = values[i]; i += 1
    gps1_lon_e7 = values[i]; i += 1
    gps1_alt_cm = values[i]; i += 1
    gps1_speed_ck = values[i]; i += 1
    gps1_time = values[i]; i += 1
    gps1_date = values[i]; i += 1
    gps2_fix = values[i]; i += 1
    gps2_sats = values[i]; i += 1
    gps2_lat_e7 = values[i]; i += 1
    gps2_lon_e7 = values[i]; i += 1
    gps2_alt_cm = values[i]; i += 1
    gps2_speed_ck = values[i]; i += 1
    gps2_time = values[i]; i += 1
    gps2_date = values[i]; i += 1
    attitude_valid = values[i]; i += 1
    q0_1e9 = values[i]; i += 1
    q1_1e9 = values[i]; i += 1
    q2_1e9 = values[i]; i += 1
    q3_1e9 = values[i]; i += 1
    roll_cdeg = values[i]; i += 1
    pitch_cdeg = values[i]; i += 1
    yaw_cdeg = values[i]; i += 1
    gyro_bias_x_cdegps = values[i]; i += 1
    gyro_bias_y_cdegps = values[i]; i += 1
    gyro_bias_z_cdegps = values[i]; i += 1
    nav = empty_nav_fields()
    if is_v2:
        nav_valid = values[i]; i += 1
        nav_gps_source = values[i]; i += 1
        nav_x_cm = values[i]; i += 1
        nav_y_cm = values[i]; i += 1
        nav_z_cm = values[i]; i += 1
        nav_vx_cms = values[i]; i += 1
        nav_vy_cms = values[i]; i += 1
        nav_vz_cms = values[i]; i += 1
        nav_ax_cms2 = values[i]; i += 1
        nav_ay_cms2 = values[i]; i += 1
        nav_az_cms2 = values[i]; i += 1
        nav_lat_e7 = values[i]; i += 1
        nav_lon_e7 = values[i]; i += 1
        nav_alt_cm = values[i]; i += 1
        nav = {
            "nav_valid": nav_valid,
            "nav_gps_source": nav_gps_source,
            "nav_x_m": nav_x_cm / 100.0,
            "nav_y_m": nav_y_cm / 100.0,
            "nav_z_m": nav_z_cm / 100.0,
            "nav_vx_mps": nav_vx_cms / 100.0,
            "nav_vy_mps": nav_vy_cms / 100.0,
            "nav_vz_mps": nav_vz_cms / 100.0,
            "nav_ax_mps2": nav_ax_cms2 / 100.0,
            "nav_ay_mps2": nav_ay_cms2 / 100.0,
            "nav_az_mps2": nav_az_cms2 / 100.0,
            "nav_lat_deg": nav_lat_e7 / 1.0e7,
            "nav_lon_deg": nav_lon_e7 / 1.0e7,
            "nav_alt_m": nav_alt_cm / 100.0,
        }
    crc_recorded = values[i]; i += 1
    crc_computed = crc16_ccitt(raw[:-2])
    crc_ok = crc_recorded == crc_computed
    row = {
        "seq": seq,
        "ms": ms,
        "time_s": ms / 1000.0,
        "icm_ax_g": icm_ax_mg / 1000.0,
        "icm_ay_g": icm_ay_mg / 1000.0,
        "icm_az_g": icm_az_mg / 1000.0,
        "icm_gx_dps": icm_gx_cdegps / 100.0,
        "icm_gy_dps": icm_gy_cdegps / 100.0,
        "icm_gz_dps": icm_gz_cdegps / 100.0,
        "adxl_ax_g": adxl_ax_cg / 100.0,
        "adxl_ay_g": adxl_ay_cg / 100.0,
        "adxl_az_g": adxl_az_cg / 100.0,
        "bmp_temp_c": bmp_temp_cC / 100.0,
        "bmp_press_pa": bmp_press_pa,
        "bmp_alt_m": bmp_alt_cm / 100.0,
        "gps1_fix": gps1_fix,
        "gps1_sats": gps1_sats,
        "gps1_lat_deg": gps1_lat_e7 / 1.0e7,
        "gps1_lon_deg": gps1_lon_e7 / 1.0e7,
        "gps1_alt_m": gps1_alt_cm / 100.0,
        "gps1_speed_knots": gps1_speed_ck / 100.0,
        "gps1_time_utc": parse_nmea_u32(gps1_time),
        "gps1_date_utc": parse_nmea_u32(gps1_date),
        "gps2_fix": gps2_fix,
        "gps2_sats": gps2_sats,
        "gps2_lat_deg": gps2_lat_e7 / 1.0e7,
        "gps2_lon_deg": gps2_lon_e7 / 1.0e7,
        "gps2_alt_m": gps2_alt_cm / 100.0,
        "gps2_speed_knots": gps2_speed_ck / 100.0,
        "gps2_time_utc": parse_nmea_u32(gps2_time),
        "gps2_date_utc": parse_nmea_u32(gps2_date),
        "attitude_valid": attitude_valid,
        "q0": q0_1e9 / 1.0e9,
        "q1": q1_1e9 / 1.0e9,
        "q2": q2_1e9 / 1.0e9,
        "q3": q3_1e9 / 1.0e9,
        "roll_deg": roll_cdeg / 100.0,
        "pitch_deg": pitch_cdeg / 100.0,
        "yaw_deg": yaw_cdeg / 100.0,
        "gyro_bias_x_dps": gyro_bias_x_cdegps / 100.0,
        "gyro_bias_y_dps": gyro_bias_y_cdegps / 100.0,
        "gyro_bias_z_dps": gyro_bias_z_cdegps / 100.0,
        "crc16_recorded": crc_recorded,
        "crc16_computed": crc_computed,
        "crc_ok": crc_ok,
    }
    row.update(nav)
    return row

def read_log(path: Path) -> tuple[Header, list[dict]]:
    with path.open("rb") as f:
        header = read_header(f)
        if header.magic != b"FLT1":
            raise ValueError(f"Bad magic. Expected b'FLT1', got {header.magic!r}.")
        if header.record_size not in (RECORD_STRUCT_V1.size, RECORD_STRUCT_V2.size):
            raise ValueError(
                "Record size mismatch.\n"
                f"Header says: {header.record_size} bytes\n"
                f"Python V1 expects: {RECORD_STRUCT_V1.size} bytes\n"
                f"Python V2 expects: {RECORD_STRUCT_V2.size} bytes\n"
                "This means the C++ LogRecord and Python struct do not match."
            )
        if header.header_size != HEADER_STRUCT.size:
            f.seek(header.header_size)
        records = []
        while True:
            raw = f.read(header.record_size)
            if not raw:
                break
            if len(raw) != header.record_size:
                print(f"Warning: ignoring trailing partial record of {len(raw)} bytes.")
                break
            records.append(decode_record(raw, header.record_size))
    return header, records

def write_csv(records: Iterable[dict], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def rows_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    data = {}
    for key in CSV_COLUMNS:
        values = [row[key] for row in records]
        if key.endswith("_utc"):
            data[key] = np.array(values, dtype=object)
        else:
            data[key] = np.array(values, dtype=float)
    return data

def deg_to_rad(deg: np.ndarray | float) -> np.ndarray | float:
    return deg * np.pi / 180.0

def latlon_to_local_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    earth_radius_m = 6378137.0
    ref_lat_rad = math.radians(ref_lat_deg)
    lat_rad = deg_to_rad(lat_deg)
    lon_rad = deg_to_rad(lon_deg)
    ref_lon_rad = math.radians(ref_lon_deg)
    north_m = (lat_rad - ref_lat_rad) * earth_radius_m
    east_m = (lon_rad - ref_lon_rad) * earth_radius_m * math.cos(ref_lat_rad)
    up_m = alt_m - ref_alt_m
    return east_m, north_m, up_m

def get_plot_position(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    nav_mask = data["nav_valid"] > 0.5
    if np.any(nav_mask):
        return (
            data["nav_x_m"],
            data["nav_y_m"],
            data["nav_z_m"],
            nav_mask,
            "EKF Nav Position",
        )
    gps1_mask = data["gps1_fix"] > 0.5
    gps2_mask = data["gps2_fix"] > 0.5
    if np.any(gps1_mask):
        ref_idx = np.where(gps1_mask)[0][0]
        x, y, z = latlon_to_local_enu(
            data["gps1_lat_deg"],
            data["gps1_lon_deg"],
            data["gps1_alt_m"],
            float(data["gps1_lat_deg"][ref_idx]),
            float(data["gps1_lon_deg"][ref_idx]),
            float(data["gps1_alt_m"][ref_idx]),
        )
        return x, y, z, gps1_mask, "GPS A Position Fallback"
    if np.any(gps2_mask):
        ref_idx = np.where(gps2_mask)[0][0]
        x, y, z = latlon_to_local_enu(
            data["gps2_lat_deg"],
            data["gps2_lon_deg"],
            data["gps2_alt_m"],
            float(data["gps2_lat_deg"][ref_idx]),
            float(data["gps2_lon_deg"][ref_idx]),
            float(data["gps2_alt_m"][ref_idx]),
        )
        return x, y, z, gps2_mask, "GPS B Position Fallback"
    zeros = np.zeros_like(data["time_s"])
    mask = np.zeros_like(data["time_s"], dtype=bool)
    return zeros, zeros, zeros, mask, "No Position Available"


def get_video_realtime_timing_lines(data: dict[str, np.ndarray]) -> list[str]:
    lines: list[str] = []
    if len(data["time_s"]) < 2:
        lines.append("Not enough timestamp data to compute real-time video multiplier.\n")
        return lines
    real_duration_s = float(data["time_s"][-1] - data["time_s"][0])
    if real_duration_s <= 0.0:
        lines.append("Invalid or zero data duration. Cannot compute real-time video multiplier.\n")
        return lines
    orientation_frame_count = min(len(data["time_s"]), int(VIDEO_MAX_FRAMES))
    orientation_video_duration_s = orientation_frame_count / float(VIDEO_FPS)
    orientation_current_speed_x = real_duration_s / orientation_video_duration_s
    orientation_editor_speed_x = orientation_video_duration_s / real_duration_s
    lines.append("Video real-time sync info\n")
    lines.append("-------------------------\n")
    lines.append(f"Recorded data duration: {real_duration_s:.3f} s\n")
    lines.append(f"Position/orientation video FPS: {VIDEO_FPS}\n")
    lines.append(f"Position/orientation video frame count: {orientation_frame_count}\n")
    lines.append(f"Position/orientation video duration: {orientation_video_duration_s:.3f} s\n")
    lines.append(f"Position/orientation video currently plays at: {orientation_current_speed_x:.3f}x real-time\n")
    lines.append(f"To make position/orientation video real-time, set video speed to: {orientation_editor_speed_x:.6f}x\n")
    lines.append(f"Equivalent editor speed percent: {orientation_editor_speed_x * 100.0:.3f}%\n")
    lines.append(f"Plain language: slow down by {orientation_current_speed_x:.3f}x if the video is too fast.\n")

    if "SATELLITE_VIDEO_FPS" in globals() and "SATELLITE_VIDEO_MAX_FRAMES" in globals():
        satellite_frame_count = min(len(data["time_s"]), int(SATELLITE_VIDEO_MAX_FRAMES))
        satellite_video_duration_s = satellite_frame_count / float(SATELLITE_VIDEO_FPS)
        satellite_current_speed_x = real_duration_s / satellite_video_duration_s
        satellite_editor_speed_x = satellite_video_duration_s / real_duration_s
        lines.append("\n")
        lines.append("Satellite overlay video real-time sync info\n")
        lines.append("-------------------------------------------\n")
        lines.append(f"Satellite video FPS: {SATELLITE_VIDEO_FPS}\n")
        lines.append(f"Satellite video max frame count setting: {SATELLITE_VIDEO_MAX_FRAMES}\n")
        lines.append(f"Satellite video estimated frame count: {satellite_frame_count}\n")
        lines.append(f"Satellite video estimated duration: {satellite_video_duration_s:.3f} s\n")
        lines.append(f"Satellite video currently plays at: {satellite_current_speed_x:.3f}x real-time\n")
        lines.append(f"To make satellite video real-time, set video speed to: {satellite_editor_speed_x:.6f}x\n")
        lines.append(f"Equivalent editor speed percent: {satellite_editor_speed_x * 100.0:.3f}%\n")
        lines.append(f"Plain language: slow down by {satellite_current_speed_x:.3f}x if the video is too fast.\n")
    return lines


def save_summary(data: dict[str, np.ndarray], header: Header, out_dir: Path) -> None:
    summary_path = out_dir / "summary.txt"
    crc_ok = data["crc_ok"] > 0.5
    attitude_valid = data["attitude_valid"] > 0.5
    nav_valid = data["nav_valid"] > 0.5
    with summary_path.open("w") as f:
        f.write("Rocket binary log decode summary\n")
        f.write("================================\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Magic: {header.magic!r}\n")
        f.write(f"Version: {header.version}\n")
        f.write(f"Header size: {header.header_size} bytes\n")
        f.write(f"Record size: {header.record_size} bytes\n")
        f.write(f"Python V1 record size: {RECORD_STRUCT_V1.size} bytes\n")
        f.write(f"Python V2 record size: {RECORD_STRUCT_V2.size} bytes\n")
        f.write(f"Start ms: {header.start_ms}\n")
        f.write(f"Records: {len(data['seq'])}\n")
        if len(data["time_s"]) > 0:
            duration = data["time_s"][-1] - data["time_s"][0]
            f.write(f"Duration: {duration:.3f} s\n")
        f.write(f"CRC OK records: {int(np.sum(crc_ok))} / {len(crc_ok)}\n")
        f.write(f"Attitude valid records: {int(np.sum(attitude_valid))} / {len(attitude_valid)}\n")
        f.write(f"Navigation valid records: {int(np.sum(nav_valid))} / {len(nav_valid)}\n")
        if len(data["roll_deg"]) > 0:
            f.write("\nEuler angle ranges\n")
            f.write("------------------\n")
            f.write(f"Roll:  {np.nanmin(data['roll_deg']):.2f} to {np.nanmax(data['roll_deg']):.2f} deg\n")
            f.write(f"Pitch: {np.nanmin(data['pitch_deg']):.2f} to {np.nanmax(data['pitch_deg']):.2f} deg\n")
            f.write(f"Yaw:   {np.nanmin(data['yaw_deg']):.2f} to {np.nanmax(data['yaw_deg']):.2f} deg\n")
        f.write("\n")
        for line in get_video_realtime_timing_lines(data):
            f.write(line)

def save_orientation_graphs(data: dict[str, np.ndarray], out_dir: Path) -> None:
    t = data["time_s"]
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, data["roll_deg"], label="Roll")
    plt.plot(t, data["pitch_deg"], label="Pitch")
    plt.plot(t, data["yaw_deg"], label="Yaw")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Orientation Euler Angles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "orientation_euler.png", dpi=200)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, data["q0"], label="q0")
    plt.plot(t, data["q1"], label="q1")
    plt.plot(t, data["q2"], label="q2")
    plt.plot(t, data["q3"], label="q3")
    plt.xlabel("Time [s]")
    plt.ylabel("Quaternion Component")
    plt.title("Orientation Quaternion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "orientation_quaternion.png", dpi=200)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, data["icm_gx_dps"], label="Gyro X")
    plt.plot(t, data["icm_gy_dps"], label="Gyro Y")
    plt.plot(t, data["icm_gz_dps"], label="Gyro Z")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Rate [deg/s]")
    plt.title("ICM20948 Gyro Rates")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "gyro_rates.png", dpi=200)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, data["icm_ax_g"], label="ICM ax")
    plt.plot(t, data["icm_ay_g"], label="ICM ay")
    plt.plot(t, data["icm_az_g"], label="ICM az")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    plt.title("ICM20948 Acceleration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "icm_accel.png", dpi=200)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, data["bmp_alt_m"], label="BMP altitude")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("BMP390 Altitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "bmp_altitude.png", dpi=200)
    plt.close(fig)

def save_position_graphs(data: dict[str, np.ndarray], out_dir: Path) -> None:
    t = data["time_s"]
    x, y, z, mask, source_name = get_plot_position(data)
    if not np.any(mask):
        print("No valid position data found. Skipping position graphs.")
        return
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, x, label="East / X")
    plt.plot(t, y, label="North / Y")
    plt.plot(t, z, label="Up / Z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title(f"Position vs Time ({source_name})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "position_xyz.png", dpi=200)
    plt.close(fig)
    if np.any(data["nav_valid"] > 0.5):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(t, data["nav_vx_mps"], label="Vx East")
        plt.plot(t, data["nav_vy_mps"], label="Vy North")
        plt.plot(t, data["nav_vz_mps"], label="Vz Up")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title("EKF Velocity vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "velocity_xyz.png", dpi=200)
        plt.close(fig)
        fig = plt.figure(figsize=(10, 4))
        plt.step(t, data["nav_gps_source"], where="post")
        plt.xlabel("Time [s]")
        plt.ylabel("GPS Source")
        plt.yticks([0, 1, 2], ["None", "GPS A", "GPS B"])
        plt.title("Active GPS Source Used by EKF")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(out_dir / "nav_gps_source.png", dpi=200)
        plt.close(fig)
    valid_idx = np.where(mask)[0]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x[valid_idx], y[valid_idx], z[valid_idx], color=TRAJECTORY_COLOR, linewidth=TRAJECTORY_LINE_WIDTH)
    ax.scatter([x[valid_idx[0]]], [y[valid_idx[0]]], [z[valid_idx[0]]], label="Start")
    ax.scatter([x[valid_idx[-1]]], [y[valid_idx[-1]]], [z[valid_idx[-1]]], label="End")
    setup_position_orientation_axes(ax, x[valid_idx], y[valid_idx], z[valid_idx], f"3D Trajectory ({source_name})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_dir / "trajectory_3d.png", dpi=200)
    plt.close(fig)

def quat_to_rotation_matrix(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
    n = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    if n <= 0.0:
        return np.eye(3)
    w = q0 / n
    x = q1 / n
    y = q2 / n
    z = q3 / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ])

def get_nose_body_vector() -> np.ndarray:
    if ROCKET_NOSE_AXIS == "+X":
        return np.array([1.0, 0.0, 0.0])
    if ROCKET_NOSE_AXIS == "-X":
        return np.array([-1.0, 0.0, 0.0])
    if ROCKET_NOSE_AXIS == "+Y":
        return np.array([0.0, 1.0, 0.0])
    if ROCKET_NOSE_AXIS == "-Y":
        return np.array([0.0, -1.0, 0.0])
    if ROCKET_NOSE_AXIS == "+Z":
        return np.array([0.0, 0.0, 1.0])
    if ROCKET_NOSE_AXIS == "-Z":
        return np.array([0.0, 0.0, -1.0])
    raise ValueError(f"Invalid ROCKET_NOSE_AXIS: {ROCKET_NOSE_AXIS}")

def get_body_and_nose_vectors(q0: float, q1: float, q2: float, q3: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = quat_to_rotation_matrix(q0, q1, q2, q3)
    body_x = R @ np.array([1.0, 0.0, 0.0])
    body_y = R @ np.array([0.0, 1.0, 0.0])
    body_z = R @ np.array([0.0, 0.0, 1.0])
    nose = R @ get_nose_body_vector()

    return body_x, body_y, body_z, nose

def compute_nose_vectors(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data["q0"])
    nose_x = np.zeros(n)
    nose_y = np.zeros(n)
    nose_z = np.zeros(n)
    for i in range(n):
        _, _, _, nose = get_body_and_nose_vectors(
            data["q0"][i],
            data["q1"][i],
            data["q2"][i],
            data["q3"][i],
        )
        nose_x[i] = nose[0]
        nose_y[i] = nose[1]
        nose_z[i] = nose[2]
    return nose_x, nose_y, nose_z

def save_rocket_nose_graphs(data: dict[str, np.ndarray], out_dir: Path) -> None:
    t = data["time_s"]
    nose_x, nose_y, nose_z = compute_nose_vectors(data)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, nose_x, label="Nose X")
    plt.plot(t, nose_y, label="Nose Y")
    plt.plot(t, nose_z, label="Nose Z")
    plt.xlabel("Time [s]")
    plt.ylabel("Unit Vector Component")
    plt.title(f"Rocket Nose Direction in World Frame ({ROCKET_NOSE_AXIS} body axis)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "rocket_nose_direction.png", dpi=200)
    plt.close(fig)

def plot_vector_line_at_position(
    ax,
    position: np.ndarray,
    vector: np.ndarray,
    length: float,
    color: str,
    linewidth: float,
    label: str | None = None,
    text: str | None = None,
) -> None:
    v = np.asarray(vector, dtype=float)
    p = np.asarray(position, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        return
    v = v / norm
    end = p + v * length
    text_pos = end + v * (AXIS_LABEL_OFFSET_FRACTION * length)
    ax.plot(
        [p[0], end[0]],
        [p[1], end[1]],
        [p[2], end[2]],
        color=color,
        linewidth=linewidth,
        label=label,
        solid_capstyle="round",
        zorder=10,
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        [end[2]],
        color=color,
        s=AXIS_TIP_MARKER_SIZE,
        depthshade=False,
        zorder=11,
    )
    if text is not None:
        ax.text(
            text_pos[0],
            text_pos[1],
            text_pos[2],
            text,
            color=color,
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.72, boxstyle="round,pad=0.16"),
            zorder=12,
        )

def plot_body_and_nose_lines_at_position(
    ax,
    position: np.ndarray,
    body_x: np.ndarray,
    body_y: np.ndarray,
    body_z: np.ndarray,
    nose: np.ndarray,
    include_labels: bool = True,
) -> None:
    p = np.asarray(position, dtype=float)
    ax.scatter(
        [p[0]],
        [p[1]],
        [p[2]],
        color=CURRENT_POSITION_COLOR,
        s=CENTER_MARKER_SIZE,
        depthshade=False,
        label="Rocket Position" if include_labels else None,
        zorder=20,
    )
    if SHOW_NEGATIVE_BODY_AXES:
        for vec, color in [
            (body_x, BODY_X_COLOR),
            (body_y, BODY_Y_COLOR),
            (body_z, BODY_Z_COLOR),
        ]:
            v = np.asarray(vec, dtype=float)
            n = np.linalg.norm(v)
            if n > 0.0:
                v = v / n
                end = p - v * (0.65 * BODY_AXIS_LENGTH_M)
                ax.plot(
                    [p[0], end[0]],
                    [p[1], end[1]],
                    [p[2], end[2]],
                    color=color,
                    linewidth=max(1.5, BODY_LINE_WIDTH * 0.55),
                    alpha=NEGATIVE_AXIS_ALPHA,
                    linestyle="--",
                    solid_capstyle="round",
                    zorder=8,
                )
    plot_vector_line_at_position(
        ax,
        position,
        body_x,
        BODY_AXIS_LENGTH_M,
        BODY_X_COLOR,
        BODY_LINE_WIDTH,
        "+Body X" if include_labels else None,
        "+X" if include_labels else None,
    )
    plot_vector_line_at_position(
        ax,
        position,
        body_y,
        BODY_AXIS_LENGTH_M,
        BODY_Y_COLOR,
        BODY_LINE_WIDTH,
        "+Body Y" if include_labels else None,
        "+Y" if include_labels else None,
    )
    plot_vector_line_at_position(
        ax,
        position,
        body_z,
        BODY_AXIS_LENGTH_M,
        BODY_Z_COLOR,
        BODY_LINE_WIDTH,
        "+Body Z" if include_labels else None,
        "+Z" if include_labels else None,
    )
    plot_vector_line_at_position(
        ax,
        position,
        nose,
        NOSE_AXIS_LENGTH_M,
        ROCKET_NOSE_COLOR,
        NOSE_LINE_WIDTH,
        "Rocket Nose" if include_labels else None,
        "NOSE" if include_labels else None,
    )

def setup_position_orientation_axes(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("East / X [m]")
    ax.set_ylabel("North / Y [m]")
    ax.set_zlabel("Up / Z [m]")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        return
    x_f = x[finite]
    y_f = y[finite]
    z_f = z[finite]
    xmin, xmax = float(np.min(x_f)), float(np.max(x_f))
    ymin, ymax = float(np.min(y_f)), float(np.max(y_f))
    zmin, zmax = float(np.min(z_f)), float(np.max(z_f))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 2.0 * NOSE_AXIS_LENGTH_M, 1.0)
    span *= 1.25
    ax.set_xlim([cx - span / 2, cx + span / 2])
    ax.set_ylim([cy - span / 2, cy + span / 2])
    ax.set_zlim([cz - span / 2, cz + span / 2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=ROCKET_AXIS_ELEV, azim=ROCKET_AXIS_AZIM)


def set_visual_axis_lengths_from_data(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    global BODY_AXIS_LENGTH_M, NOSE_AXIS_LENGTH_M
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        BODY_AXIS_LENGTH_M = 2.0
        NOSE_AXIS_LENGTH_M = 2.5
        return
    x_f = x[finite]
    y_f = y[finite]
    z_f = z[finite]
    span = max(
        float(np.max(x_f) - np.min(x_f)),
        float(np.max(y_f) - np.min(y_f)),
        float(np.max(z_f) - np.min(z_f)),
        1.0,
    )
    BODY_AXIS_LENGTH_M = max(2.0, AXIS_VISUAL_FRACTION_OF_PLOT * span)
    NOSE_AXIS_LENGTH_M = max(2.5, 1.25 * BODY_AXIS_LENGTH_M)


def _gps_path_from_columns(
    data: dict[str, np.ndarray],
    lat_col: str,
    lon_col: str,
    valid_mask: np.ndarray,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    lat = data[lat_col]
    lon = data[lon_col]
    mask = (
        valid_mask &
        np.isfinite(lat) &
        np.isfinite(lon) &
        (np.abs(lat) > 0.000001) &
        (np.abs(lon) > 0.000001)
    )
    idx = np.where(mask)[0]
    points = [(float(lat[i]), float(lon[i])) for i in idx]
    return points, idx

def _add_path_to_map(
    m,
    points: list[tuple[float, float]],
    times: np.ndarray,
    name: str,
    weight: int = 5,
) -> None:
    if len(points) < 1:
        return
    folium.PolyLine(
        points,
        weight=weight,
        opacity=0.90,
        tooltip=name,
        popup=name,
    ).add_to(m)
    folium.Marker(
        points[0],
        tooltip=f"{name} Start",
        popup=(
            f"{name} Start<br>"
            f"t={float(times[0]):.2f}s<br>"
            f"lat={points[0][0]:.8f}<br>"
            f"lon={points[0][1]:.8f}"
        ),
    ).add_to(m)
    folium.Marker(
        points[-1],
        tooltip=f"{name} End",
        popup=(
            f"{name} End<br>"
            f"t={float(times[-1]):.2f}s<br>"
            f"lat={points[-1][0]:.8f}<br>"
            f"lon={points[-1][1]:.8f}"
        ),
    ).add_to(m)

def save_satellite_gps_map(data: dict[str, np.ndarray], out_dir: Path) -> None:
    if folium is None:
        print("Folium is not installed. Skipping satellite map.")
        print("Install it with: pip install folium")
        return
    gps1_points, gps1_idx = _gps_path_from_columns(
        data,
        "gps1_lat_deg",
        "gps1_lon_deg",
        data["gps1_fix"] > 0.5,
    )
    gps2_points, gps2_idx = _gps_path_from_columns(
        data,
        "gps2_lat_deg",
        "gps2_lon_deg",
        data["gps2_fix"] > 0.5,
    )
    nav_points, nav_idx = _gps_path_from_columns(
        data,
        "nav_lat_deg",
        "nav_lon_deg",
        data["nav_valid"] > 0.5,
    )
    all_points = []
    all_points.extend(nav_points)
    all_points.extend(gps1_points)
    all_points.extend(gps2_points)
    if len(all_points) < 1:
        print("No valid GPS latitude/longitude found. Skipping satellite map.")
        return
    center_lat = float(np.mean([p[0] for p in all_points]))
    center_lon = float(np.mean([p[1] for p in all_points]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)
    if len(nav_points) > 0:
        _add_path_to_map(m, nav_points, data["time_s"][nav_idx], "EKF Nav Path", weight=6)
    if len(gps1_points) > 0:
        _add_path_to_map(m, gps1_points, data["time_s"][gps1_idx], "GPS A Path", weight=4)
    if len(gps2_points) > 0:
        _add_path_to_map(m, gps2_points, data["time_s"][gps2_idx], "GPS B Path", weight=4)
    folium.LayerControl().add_to(m)
    m.fit_bounds(all_points)
    map_path = out_dir / "satellite_gps_path.html"
    m.save(map_path)
    print(f"Satellite GPS map: {map_path}")

def _choose_best_gps_animation_path(data: dict[str, np.ndarray]) -> tuple[list[tuple[float, float]], np.ndarray, str]:
    candidates = []
    nav_points, nav_idx = _gps_path_from_columns(
        data,
        "nav_lat_deg",
        "nav_lon_deg",
        data["nav_valid"] > 0.5,
    )
    candidates.append((nav_points, nav_idx, "EKF Nav Path"))
    gps1_points, gps1_idx = _gps_path_from_columns(
        data,
        "gps1_lat_deg",
        "gps1_lon_deg",
        data["gps1_fix"] > 0.5,
    )
    candidates.append((gps1_points, gps1_idx, "GPS A Path"))
    gps2_points, gps2_idx = _gps_path_from_columns(
        data,
        "gps2_lat_deg",
        "gps2_lon_deg",
        data["gps2_fix"] > 0.5,
    )
    candidates.append((gps2_points, gps2_idx, "GPS B Path"))
    for points, idx, name in candidates:
        if len(points) >= 2:
            return points, idx, name
    return [], np.array([], dtype=int), "No GPS Path"

def _thin_animation_points(
    points: list[tuple[float, float]],
    idx: np.ndarray,
    max_points: int,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    if len(points) <= max_points:
        return points, idx
    selected = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[i] for i in selected], idx[selected]

def _json_number_array(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.10g}" for v in values) + "]"

def _json_points_array(points: list[tuple[float, float]]) -> str:
    return "[" + ",".join(f"[{float(lat):.10f},{float(lon):.10f}]" for lat, lon in points) + "]"

def _map_bounds_js(points: list[tuple[float, float]]) -> str:
    return "[" + ",".join(f"[{float(lat):.10f},{float(lon):.10f}]" for lat, lon in points) + "]"

def save_satellite_gps_animation(data: dict[str, np.ndarray], out_dir: Path) -> None:
    points, idx, source_name = _choose_best_gps_animation_path(data)
    if len(points) < 2:
        print("Not enough GPS points for satellite animation. Skipping animated map.")
        return
    points, idx = _thin_animation_points(points, idx, SATELLITE_ANIMATION_MAX_POINTS)
    times = [float(data["time_s"][i]) for i in idx]
    center_lat = float(np.mean([p[0] for p in points]))
    center_lon = float(np.mean([p[1] for p in points]))
    points_js = _json_points_array(points)
    times_js = _json_number_array(times)
    bounds_js = _map_bounds_js(points)
    dt_ms = int(round(1000.0 / max(float(SATELLITE_ANIMATION_FPS), 1.0)))
    playback_speed = float(SATELLITE_ANIMATION_PLAYBACK_SPEED)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Satellite GPS Animation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>

    <style>
        html, body {{
            height: 100%;
            margin: 0;
            font-family: Arial, sans-serif;
        }}

        #map {{
            height: 100%;
            width: 100%;
        }}

        .control-panel {{
            position: absolute;
            left: 16px;
            bottom: 18px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.92);
            padding: 12px 14px;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
            min-width: 330px;
        }}

        .control-row {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        button {{
            font-size: 14px;
            padding: 6px 10px;
            cursor: pointer;
        }}

        #slider {{
            width: 230px;
        }}

        .readout {{
            margin-top: 8px;
            font-size: 13px;
            line-height: 1.35;
        }}

        .legend {{
            position: absolute;
            right: 16px;
            bottom: 18px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.92);
            padding: 10px 12px;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
            font-size: 13px;
        }}

        .dot {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 6px;
            margin-right: 6px;
            vertical-align: middle;
        }}

        .rocket-marker {{
            width: 20px;
            height: 20px;
            background: #ff7a00;
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 0 10px rgba(0,0,0,0.8);
        }}
    </style>
</head>

<body>
    <div id="map"></div>

    <div class="control-panel">
        <div class="control-row">
            <button id="playPause">Pause</button>
            <input id="slider" type="range" min="0" max="{len(points) - 1}" value="0" step="1">
            <button id="restart">Restart</button>
        </div>
        <div class="readout">
            <div><b>Source:</b> {source_name}</div>
            <div><b>Time:</b> <span id="timeReadout">0.00</span> s</div>
            <div><b>Lat/Lon:</b> <span id="latlonReadout"></span></div>
            <div><b>Frame:</b> <span id="frameReadout">0</span> / {len(points) - 1}</div>
        </div>
    </div>

    <div class="legend">
        <div><span class="dot" style="background:#00ffff"></span>Full GPS path</div>
        <div><span class="dot" style="background:#ff7a00"></span>Current rocket position</div>
    </div>

    <script>
        const points = {points_js};
        const times = {times_js};
        const bounds = {bounds_js};

        const map = L.map('map', {{
            preferCanvas: true
        }}).setView([{center_lat:.10f}, {center_lon:.10f}], 17);

        const satellite = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{
                attribution: 'Tiles © Esri, Maxar, Earthstar Geographics, and the GIS User Community',
                maxZoom: 20
            }}
        ).addTo(map);

        const osm = L.tileLayer(
            'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
            {{
                attribution: '&copy; OpenStreetMap contributors',
                maxZoom: 19
            }}
        );

        const fullPath = L.polyline(points, {{
            color: '#00ffff',
            weight: 5,
            opacity: 0.85
        }}).addTo(map);

        L.marker(points[0]).addTo(map).bindPopup('Start');
        L.marker(points[points.length - 1]).addTo(map).bindPopup('End');

        const rocketIcon = L.divIcon({{
            className: '',
            html: '<div class="rocket-marker"></div>',
            iconSize: [26, 26],
            iconAnchor: [13, 13]
        }});

        const rocket = L.marker(points[0], {{ icon: rocketIcon }}).addTo(map);

        L.control.layers({{
            "Satellite": satellite,
            "OpenStreetMap": osm
        }}, {{
            "Full path": fullPath,
            "Rocket": rocket
        }}).addTo(map);

        map.fitBounds(bounds, {{ padding: [35, 35] }});

        const slider = document.getElementById('slider');
        const playPause = document.getElementById('playPause');
        const restart = document.getElementById('restart');
        const timeReadout = document.getElementById('timeReadout');
        const latlonReadout = document.getElementById('latlonReadout');
        const frameReadout = document.getElementById('frameReadout');

        let frame = 0;
        let playing = true;
        let lastTick = performance.now();

        function updateFrame(i) {{
            frame = Math.max(0, Math.min(points.length - 1, i));

            const p = points[frame];
            rocket.setLatLng(p);

            slider.value = frame;
            timeReadout.textContent = times[frame].toFixed(2);
            latlonReadout.textContent = p[0].toFixed(8) + ', ' + p[1].toFixed(8);
            frameReadout.textContent = frame.toString();
        }}

        function stepAnimation(now) {{
            if (playing && now - lastTick >= {dt_ms}) {{
                lastTick = now;

                let nextFrame = frame + Math.max(1, Math.round({playback_speed}));

                if (nextFrame >= points.length) {{
                    nextFrame = 0;
                }}

                updateFrame(nextFrame);
            }}

            requestAnimationFrame(stepAnimation);
        }}

        playPause.addEventListener('click', () => {{
            playing = !playing;
            playPause.textContent = playing ? 'Pause' : 'Play';
        }});

        restart.addEventListener('click', () => {{
            updateFrame(0);
            playing = true;
            playPause.textContent = 'Pause';
        }});

        slider.addEventListener('input', () => {{
            playing = false;
            playPause.textContent = 'Play';
            updateFrame(parseInt(slider.value));
        }});

        updateFrame(0);
        requestAnimationFrame(stepAnimation);
    </script>
</body>
</html>
"""
    animation_path = out_dir / "satellite_gps_animation.html"
    animation_path.write_text(html, encoding="utf-8")
    print(f"Satellite GPS animation: {animation_path}")

def _latlon_to_tile_float(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (
        1.0 -
        math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi
    ) / 2.0 * n
    return x, y

def _choose_satellite_zoom(points: list[tuple[float, float]]) -> tuple[int, int, int, int, int]:
    for zoom in range(SATELLITE_VIDEO_MAX_ZOOM, SATELLITE_VIDEO_MIN_ZOOM - 1, -1):
        xs = []
        ys = []
        for lat, lon in points:
            x, y = _latlon_to_tile_float(lat, lon, zoom)
            xs.append(x)
            ys.append(y)
        min_x = max(0, int(math.floor(min(xs))) - 1)
        max_x = int(math.floor(max(xs))) + 1
        min_y = max(0, int(math.floor(min(ys))) - 1)
        max_y = int(math.floor(max(ys))) + 1
        tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)
        if tile_count <= SATELLITE_VIDEO_MAX_TILES:
            return zoom, min_x, max_x, min_y, max_y
    zoom = SATELLITE_VIDEO_MIN_ZOOM
    xs = []
    ys = []
    for lat, lon in points:
        x, y = _latlon_to_tile_float(lat, lon, zoom)
        xs.append(x)
        ys.append(y)
    return (
        zoom,
        max(0, int(math.floor(min(xs))) - 1),
        int(math.floor(max(xs))) + 1,
        max(0, int(math.floor(min(ys))) - 1),
        int(math.floor(max(ys))) + 1,
    )

def _download_esri_tile(zoom: int, x: int, y: int):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _build_satellite_mosaic(points: list[tuple[float, float]]):
    if requests is None or Image is None or BytesIO is None:
        raise ImportError("Missing requests/Pillow. Install with: pip install requests pillow")
    zoom, min_x, max_x, min_y, max_y = _choose_satellite_zoom(points)
    tile_size = 256
    nx = max_x - min_x + 1
    ny = max_y - min_y + 1
    mosaic = Image.new("RGB", (nx * tile_size, ny * tile_size), color=(235, 235, 235))
    print(f"Downloading satellite tiles: zoom={zoom}, tiles={nx * ny}")
    for tx in range(min_x, max_x + 1):
        for ty in range(min_y, max_y + 1):
            try:
                tile = _download_esri_tile(zoom, tx, ty)
            except Exception as e:
                print(f"Warning: failed satellite tile z={zoom}, x={tx}, y={ty}: {e}")
                continue
            px = (tx - min_x) * tile_size
            py = (ty - min_y) * tile_size
            mosaic.paste(tile, (px, py))
    pixel_points = []
    for lat, lon in points:
        xf, yf = _latlon_to_tile_float(lat, lon, zoom)
        px = (xf - min_x) * tile_size
        py = (yf - min_y) * tile_size
        pixel_points.append((px, py))
    return np.asarray(mosaic), np.array(pixel_points, dtype=float)

def save_satellite_gps_video(data: dict[str, np.ndarray], out_dir: Path) -> None:
    points, idx, source_name = _choose_best_gps_animation_path(data)
    if len(points) < 2:
        print("Not enough GPS points for satellite video. Skipping satellite video.")
        return
    points, idx = _thin_animation_points(points, idx, SATELLITE_VIDEO_MAX_FRAMES)
    times = [float(data["time_s"][i]) for i in idx]
    try:
        satellite_img, pixel_points = _build_satellite_mosaic(points)
    except Exception as e:
        print(f"Could not build satellite video background. Reason: {e}")
        return
    h, w = satellite_img.shape[:2]
    fig_w = 12
    fig_h = fig_w * h / max(w, 1)
    fig = plt.figure(figsize=(fig_w, fig_h), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(satellite_img, aspect="auto")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.plot(
        pixel_points[:, 0],
        pixel_points[:, 1],
        linewidth=3.0,
        color="cyan",
        label=source_name,
    )
    rocket_dot, = ax.plot(
        [pixel_points[0, 0]],
        [pixel_points[0, 1]],
        marker="o",
        markersize=12,
        color="orange",
        markeredgecolor="white",
        markeredgewidth=2,
        linestyle="None",
        label="Rocket",
    )
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor="black", alpha=0.65, boxstyle="round,pad=0.35"),
    )
    ax.legend(loc="lower right")
    def draw_frame(frame_number: int):
        p = pixel_points[frame_number]
        rocket_dot.set_data([p[0]], [p[1]])
        time_text.set_text(f"EKF Navigation Path Test 66 (CRG Stoneham to Market Basket), t = {times[frame_number]:.2f} s")
        return [rocket_dot, time_text]
    animation = FuncAnimation(
        fig,
        draw_frame,
        frames=len(pixel_points),
        interval=1000 / max(SATELLITE_VIDEO_FPS, 1),
        blit=False,
    )
    mp4_path = out_dir / "satellite_gps_animation.mp4"
    gif_path = out_dir / "satellite_gps_animation.gif"
    try:
        writer = FFMpegWriter(fps=SATELLITE_VIDEO_FPS)
        animation.save(mp4_path, writer=writer, dpi=SATELLITE_VIDEO_DPI)
        print(f"Satellite GPS video: {mp4_path}")
    except Exception as e:
        print(f"Could not save satellite MP4, falling back to GIF. Reason: {e}")
        writer = PillowWriter(fps=SATELLITE_VIDEO_FPS)
        animation.save(gif_path, writer=writer, dpi=SATELLITE_VIDEO_DPI)
        print(f"Satellite GPS GIF: {gif_path}")
    plt.close(fig)

def save_position_orientation_final_3d(data: dict[str, np.ndarray], out_dir: Path) -> None:
    x, y, z, pos_mask, source_name = get_plot_position(data)
    att_mask = data["attitude_valid"] > 0.5
    valid = pos_mask & att_mask
    if not np.any(valid):
        print("No combined position + attitude records found. Skipping combined final plot.")
        return
    valid_idx = np.where(valid)[0]
    set_visual_axis_lengths_from_data(x[valid_idx], y[valid_idx], z[valid_idx])
    idx = valid_idx[-1]
    position = np.array([x[idx], y[idx], z[idx]])
    body_x, body_y, body_z, nose = get_body_and_nose_vectors(
        data["q0"][idx],
        data["q1"][idx],
        data["q2"][idx],
        data["q3"][idx],
    )
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x[valid_idx], y[valid_idx], z[valid_idx], color=TRAJECTORY_COLOR, linewidth=TRAJECTORY_LINE_WIDTH, label="Trajectory")
    plot_body_and_nose_lines_at_position(ax, position, body_x, body_y, body_z, nose)
    setup_position_orientation_axes(
        ax,
        x[valid_idx],
        y[valid_idx],
        z[valid_idx],
        f"Position + Orientation, t = {data['time_s'][idx]:.2f} s ({source_name})",
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_dir / "position_orientation_final_3d.png", dpi=200)
    plt.close(fig)

def save_position_orientation_frames(data: dict[str, np.ndarray], out_dir: Path) -> None:
    x, y, z, pos_mask, source_name = get_plot_position(data)
    att_mask = data["attitude_valid"] > 0.5
    valid = pos_mask & att_mask
    if not np.any(valid):
        print("No combined position + attitude records found. Skipping combined frames.")
        return
    valid_idx = np.where(valid)[0]
    set_visual_axis_lengths_from_data(x[valid_idx], y[valid_idx], z[valid_idx])
    if valid_idx.size > MAX_POSITION_ORIENTATION_FRAMES:
        selected = np.linspace(0, valid_idx.size - 1, MAX_POSITION_ORIENTATION_FRAMES).astype(int)
        frame_indices = valid_idx[selected]
    else:
        frame_indices = valid_idx
    frames_dir = out_dir / "position_orientation_frames"
    frames_dir.mkdir(exist_ok=True)
    for frame_number, idx in enumerate(frame_indices):
        position = np.array([x[idx], y[idx], z[idx]])
        body_x, body_y, body_z, nose = get_body_and_nose_vectors(
            data["q0"][idx],
            data["q1"][idx],
            data["q2"][idx],
            data["q3"][idx],
        )
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        shown = valid_idx[valid_idx <= idx]
        ax.plot(x[shown], y[shown], z[shown], color=TRAJECTORY_COLOR, linewidth=TRAJECTORY_LINE_WIDTH, label="Trajectory")
        plot_body_and_nose_lines_at_position(ax, position, body_x, body_y, body_z, nose)
        setup_position_orientation_axes(
            ax,
            x[valid_idx],
            y[valid_idx],
            z[valid_idx],
            f"Position + Orientation Frame {frame_number:04d}, t = {data['time_s'][idx]:.2f} s ({source_name})",
        )
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(frames_dir / f"position_orientation_{frame_number:04d}.png", dpi=150)
        plt.close(fig)
    print(f"Saved {len(frame_indices)} position + orientation frame images to: {frames_dir}")

def save_position_orientation_video(data: dict[str, np.ndarray], out_dir: Path) -> None:
    x, y, z, pos_mask, source_name = get_plot_position(data)
    att_mask = data["attitude_valid"] > 0.5
    valid = pos_mask & att_mask
    if not np.any(valid):
        print("No combined position + attitude records found. Skipping combined video.")
        return
    valid_idx = np.where(valid)[0]
    set_visual_axis_lengths_from_data(x[valid_idx], y[valid_idx], z[valid_idx])
    if valid_idx.size > VIDEO_MAX_FRAMES:
        selected = np.linspace(0, valid_idx.size - 1, VIDEO_MAX_FRAMES).astype(int)
        frame_indices = valid_idx[selected]
    else:
        frame_indices = valid_idx
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    def draw_frame(frame_number: int):
        idx = frame_indices[frame_number]
        position = np.array([x[idx], y[idx], z[idx]])
        body_x, body_y, body_z, nose = get_body_and_nose_vectors(
            data["q0"][idx],
            data["q1"][idx],
            data["q2"][idx],
            data["q3"][idx],
        )
        shown = valid_idx[valid_idx <= idx]
        ax.cla()
        ax.plot(x[shown], y[shown], z[shown], color=TRAJECTORY_COLOR, linewidth=TRAJECTORY_LINE_WIDTH, label="Trajectory")
        plot_body_and_nose_lines_at_position(
            ax,
            position,
            body_x,
            body_y,
            body_z,
            nose,
            include_labels=True,
        )
        setup_position_orientation_axes(
            ax,
            x[valid_idx],
            y[valid_idx],
            z[valid_idx],
            f"Position + Orientation, t = {data['time_s'][idx]:.2f} s ({source_name})",
        )
        ax.legend(loc="upper right")
        return []
    animation = FuncAnimation(
        fig,
        draw_frame,
        frames=len(frame_indices),
        interval=1000 / VIDEO_FPS,
        blit=False,
    )
    mp4_path = out_dir / "position_orientation_video.mp4"
    gif_path = out_dir / "position_orientation_video.gif"
    try:
        writer = FFMpegWriter(fps=VIDEO_FPS)
        animation.save(mp4_path, writer=writer, dpi=150)
        print(f"Saved position + orientation video: {mp4_path}")
    except Exception as e:
        print(f"Could not save MP4, falling back to GIF. Reason: {e}")
        writer = PillowWriter(fps=VIDEO_FPS)
        animation.save(gif_path, writer=writer, dpi=120)
        print(f"Saved position + orientation GIF: {gif_path}")
    plt.close(fig)

def main() -> None:
    bin_path = SCRIPT_DIR / INPUT_FILE
    if not bin_path.exists():
        raise FileNotFoundError(
            f"\nCould not find this file:\n"
            f"{bin_path}\n\n"
            f"Make sure main.py and {INPUT_FILE} are in the SAME folder,\n"
            f"or change INPUT_FILE at the top of main.py."
        )
    if OUTPUT_DIR is None:
        out_dir = SCRIPT_DIR / f"{bin_path.stem}_decoded"
    else:
        out_dir = Path(OUTPUT_DIR)

        if not out_dir.is_absolute():
            out_dir = SCRIPT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input file: {bin_path}")
    print(f"Output folder: {out_dir}")
    header, records = read_log(bin_path)
    print(f"Header version: {header.version}")
    print(f"Header record size: {header.record_size} bytes")
    print(f"Python V1 record size: {RECORD_STRUCT_V1.size} bytes")
    print(f"Python V2 record size: {RECORD_STRUCT_V2.size} bytes")
    print(f"Decoded records: {len(records)}")
    if not records:
        raise ValueError("No records found in binary log.")
    csv_path = out_dir / f"{bin_path.stem}.csv"
    write_csv(records, csv_path)
    data = rows_to_arrays(records)
    save_summary(data, header, out_dir)
    bad_crc_count = int(np.sum(data["crc_ok"] < 0.5))
    if bad_crc_count > 0:
        print(f"WARNING: {bad_crc_count} records failed CRC.")
    else:
        print("CRC check: all records OK.")
    if MAKE_PLOTS:
        save_orientation_graphs(data, out_dir)
        save_rocket_nose_graphs(data, out_dir)
        save_position_graphs(data, out_dir)
        save_position_orientation_final_3d(data, out_dir)
        if MAKE_SATELLITE_GPS_MAP:
            save_satellite_gps_map(data, out_dir)
        if MAKE_SATELLITE_GPS_ANIMATION:
            save_satellite_gps_animation(data, out_dir)
        if MAKE_SATELLITE_GPS_VIDEO:
            save_satellite_gps_video(data, out_dir)
        if MAKE_POSITION_ORIENTATION_FRAMES:
            save_position_orientation_frames(data, out_dir)
        if MAKE_VIDEO and MAKE_POSITION_ORIENTATION_VIDEO:
            save_position_orientation_video(data, out_dir)
    print("")
    print("Done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {out_dir / 'summary.txt'}")
    if MAKE_PLOTS:
        print(f"Euler plot: {out_dir / 'orientation_euler.png'}")
        print(f"Quaternion plot: {out_dir / 'orientation_quaternion.png'}")
        print(f"Rocket nose plot: {out_dir / 'rocket_nose_direction.png'}")
        print(f"Position plot: {out_dir / 'position_xyz.png'}")
        print(f"Trajectory plot: {out_dir / 'trajectory_3d.png'}")
        print(f"Position + orientation plot: {out_dir / 'position_orientation_final_3d.png'}")
        if MAKE_SATELLITE_GPS_MAP:
            print(f"Satellite GPS map: {out_dir / 'satellite_gps_path.html'}")
        if MAKE_VIDEO and MAKE_POSITION_ORIENTATION_VIDEO:
            print(f"Position + orientation video: {out_dir / 'position_orientation_video.mp4'}")

if __name__ == "__main__":
    main()