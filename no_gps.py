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

INPUT_FILE = "FLIGHT00.BIN"
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
CENTER_COLOR = "orange"
BODY_LINE_WIDTH = 3.0
NOSE_LINE_WIDTH = 4.0
BODY_AXIS_LENGTH = 1.0
NOSE_AXIS_LENGTH = 1.35
AXIS_TIP_MARKER_SIZE = 65
CENTER_MARKER_SIZE = 55
AXIS_LABEL_OFFSET_FRACTION = 0.14
SHOW_NEGATIVE_BODY_AXES = True
NEGATIVE_AXIS_ALPHA = 0.28
ROCKET_AXIS_ELEV = 22
ROCKET_AXIS_AZIM = -55

SCRIPT_DIR = Path(__file__).resolve().parent
HEADER_STRUCT = struct.Struct("<4sHHHHI")
RECORD_STRUCT_V1 = struct.Struct("<" "II" "hhhhhh" "hhh" "hIi" "BBiiiHII" "BBiiiHII" "B" "iiii" "hhhhhh" "H")
RECORD_STRUCT_V2 = struct.Struct("<" "II" "hhhhhh" "hhh" "hIi" "BBiiiHII" "BBiiiHII" "B" "iiii" "hhhhhh" "BB" "iiiiiiiiiiii" "H")
CSV_COLUMNS = ["seq", "ms", "time_s", "icm_ax_g", "icm_ay_g", "icm_az_g", "icm_gx_dps", "icm_gy_dps", "icm_gz_dps", "adxl_ax_g", "adxl_ay_g", "adxl_az_g", "bmp_temp_c", "bmp_press_pa", "bmp_alt_m", "attitude_valid", "q0", "q1", "q2", "q3", "roll_deg", "pitch_deg", "yaw_deg", "gyro_bias_x_dps", "gyro_bias_y_dps", "gyro_bias_z_dps", "crc16_recorded", "crc16_computed", "crc_ok"]

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

def read_header(f) -> Header:
    raw = f.read(HEADER_STRUCT.size)
    if len(raw) != HEADER_STRUCT.size:
        raise ValueError("File is too small to contain a valid header.")
    magic, version, header_size, record_size, reserved, start_ms = HEADER_STRUCT.unpack(raw)
    return Header(magic=magic, version=version, header_size=header_size, record_size=record_size, reserved=reserved, start_ms=start_ms)

def decode_record(raw: bytes, record_size: int) -> dict:
    if record_size == RECORD_STRUCT_V1.size:
        values = RECORD_STRUCT_V1.unpack(raw)
        is_v2 = False
    elif record_size == RECORD_STRUCT_V2.size:
        values = RECORD_STRUCT_V2.unpack(raw)
        is_v2 = True
    else:
        raise ValueError(f"Unsupported record length: {record_size}. V1 expects {RECORD_STRUCT_V1.size}, V2 expects {RECORD_STRUCT_V2.size}.")
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
    i += 8
    i += 8
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
    if is_v2:
        i += 14
    crc_recorded = values[i]; i += 1
    crc_computed = crc16_ccitt(raw[:-2])
    crc_ok = crc_recorded == crc_computed
    return {"seq": seq, "ms": ms, "time_s": ms / 1000.0, "icm_ax_g": icm_ax_mg / 1000.0, "icm_ay_g": icm_ay_mg / 1000.0, "icm_az_g": icm_az_mg / 1000.0, "icm_gx_dps": icm_gx_cdegps / 100.0, "icm_gy_dps": icm_gy_cdegps / 100.0, "icm_gz_dps": icm_gz_cdegps / 100.0, "adxl_ax_g": adxl_ax_cg / 100.0, "adxl_ay_g": adxl_ay_cg / 100.0, "adxl_az_g": adxl_az_cg / 100.0, "bmp_temp_c": bmp_temp_cC / 100.0, "bmp_press_pa": bmp_press_pa, "bmp_alt_m": bmp_alt_cm / 100.0, "attitude_valid": attitude_valid, "q0": q0_1e9 / 1.0e9, "q1": q1_1e9 / 1.0e9, "q2": q2_1e9 / 1.0e9, "q3": q3_1e9 / 1.0e9, "roll_deg": roll_cdeg / 100.0, "pitch_deg": pitch_cdeg / 100.0, "yaw_deg": yaw_cdeg / 100.0, "gyro_bias_x_dps": gyro_bias_x_cdegps / 100.0, "gyro_bias_y_dps": gyro_bias_y_cdegps / 100.0, "gyro_bias_z_dps": gyro_bias_z_cdegps / 100.0, "crc16_recorded": crc_recorded, "crc16_computed": crc_computed, "crc_ok": crc_ok}

def read_log(path: Path) -> tuple[Header, list[dict]]:
    with path.open("rb") as f:
        header = read_header(f)
        if header.magic != b"FLT1":
            raise ValueError(f"Bad magic. Expected b'FLT1', got {header.magic!r}.")
        if header.record_size not in (RECORD_STRUCT_V1.size, RECORD_STRUCT_V2.size):
            raise ValueError("Record size mismatch.\n" f"Header says: {header.record_size} bytes\n" f"Python V1 expects: {RECORD_STRUCT_V1.size} bytes\n" f"Python V2 expects: {RECORD_STRUCT_V2.size} bytes\n" "This means the C++ LogRecord and Python struct do not match.")
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
        data[key] = np.array([row[key] for row in records], dtype=float)
    return data

def get_video_realtime_timing_lines(data: dict[str, np.ndarray]) -> list[str]:
    lines: list[str] = []
    if len(data["time_s"]) < 2:
        lines.append("Not enough timestamp data to compute real-time video multiplier.\n")
        return lines
    real_duration_s = float(data["time_s"][-1] - data["time_s"][0])
    if real_duration_s <= 0.0:
        lines.append("Invalid or zero data duration. Cannot compute real-time video multiplier.\n")
        return lines
    frame_count = min(int(np.sum(data["attitude_valid"] > 0.5)), int(VIDEO_MAX_FRAMES))
    video_duration_s = frame_count / float(VIDEO_FPS)
    current_speed_x = real_duration_s / video_duration_s
    editor_speed_x = video_duration_s / real_duration_s
    lines.append("Orientation video real-time sync info\n")
    lines.append("-------------------------------------\n")
    lines.append(f"Recorded data duration: {real_duration_s:.3f} s\n")
    lines.append(f"Orientation video FPS: {VIDEO_FPS}\n")
    lines.append(f"Orientation video frame count: {frame_count}\n")
    lines.append(f"Orientation video duration: {video_duration_s:.3f} s\n")
    lines.append(f"Orientation video currently plays at: {current_speed_x:.3f}x real-time\n")
    lines.append(f"To make orientation video real-time, set video speed to: {editor_speed_x:.6f}x\n")
    lines.append(f"Equivalent editor speed percent: {editor_speed_x * 100.0:.3f}%\n")
    lines.append(f"Plain language: slow down by {current_speed_x:.3f}x if the video is too fast.\n")
    return lines

def save_summary(data: dict[str, np.ndarray], header: Header, out_dir: Path) -> None:
    summary_path = out_dir / "summary.txt"
    crc_ok = data["crc_ok"] > 0.5
    attitude_valid = data["attitude_valid"] > 0.5
    with summary_path.open("w") as f:
        f.write("Rocket binary log decode summary, no GPS\n")
        f.write("========================================\n\n")
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

def quat_to_rotation_matrix(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
    n = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    if n <= 0.0:
        return np.eye(3)
    w = q0 / n
    x = q1 / n
    y = q2 / n
    z = q3 / n
    return np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]])

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
        _, _, _, nose = get_body_and_nose_vectors(data["q0"][i], data["q1"][i], data["q2"][i], data["q3"][i])
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

def plot_vector_line(ax, vector: np.ndarray, length: float, color: str, linewidth: float, label: str | None = None, text: str | None = None) -> None:
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        return
    v = v / norm
    end = v * length
    text_pos = end + v * (AXIS_LABEL_OFFSET_FRACTION * length)
    ax.plot([0.0, end[0]], [0.0, end[1]], [0.0, end[2]], color=color, linewidth=linewidth, label=label, solid_capstyle="round", zorder=10)
    ax.scatter([end[0]], [end[1]], [end[2]], color=color, s=AXIS_TIP_MARKER_SIZE, depthshade=False, zorder=11)
    if text is not None:
        ax.text(text_pos[0], text_pos[1], text_pos[2], text, color=color, fontsize=11, fontweight="bold", bbox=dict(facecolor="white", edgecolor=color, alpha=0.72, boxstyle="round,pad=0.16"), zorder=12)

def plot_body_and_nose(ax, body_x: np.ndarray, body_y: np.ndarray, body_z: np.ndarray, nose: np.ndarray, include_labels: bool = True) -> None:
    ax.scatter([0.0], [0.0], [0.0], color=CENTER_COLOR, s=CENTER_MARKER_SIZE, depthshade=False, label="Rocket Center" if include_labels else None, zorder=20)
    if SHOW_NEGATIVE_BODY_AXES:
        for vec, color in [(body_x, BODY_X_COLOR), (body_y, BODY_Y_COLOR), (body_z, BODY_Z_COLOR)]:
            v = np.asarray(vec, dtype=float)
            n = np.linalg.norm(v)
            if n > 0.0:
                v = v / n
                end = -v * (0.65 * BODY_AXIS_LENGTH)
                ax.plot([0.0, end[0]], [0.0, end[1]], [0.0, end[2]], color=color, linewidth=max(1.5, BODY_LINE_WIDTH * 0.55), alpha=NEGATIVE_AXIS_ALPHA, linestyle="--", solid_capstyle="round", zorder=8)
    plot_vector_line(ax, body_x, BODY_AXIS_LENGTH, BODY_X_COLOR, BODY_LINE_WIDTH, "+Body X" if include_labels else None, "+X" if include_labels else None)
    plot_vector_line(ax, body_y, BODY_AXIS_LENGTH, BODY_Y_COLOR, BODY_LINE_WIDTH, "+Body Y" if include_labels else None, "+Y" if include_labels else None)
    plot_vector_line(ax, body_z, BODY_AXIS_LENGTH, BODY_Z_COLOR, BODY_LINE_WIDTH, "+Body Z" if include_labels else None, "+Z" if include_labels else None)
    plot_vector_line(ax, nose, NOSE_AXIS_LENGTH, ROCKET_NOSE_COLOR, NOSE_LINE_WIDTH, "Rocket Nose" if include_labels else None, "NOSE" if include_labels else None)

def setup_orientation_axes(ax, title: str) -> None:
    span = max(1.75 * NOSE_AXIS_LENGTH, 1.5)
    ax.set_title(title)
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")
    ax.set_xlim([-span, span])
    ax.set_ylim([-span, span])
    ax.set_zlim([-span, span])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=ROCKET_AXIS_ELEV, azim=ROCKET_AXIS_AZIM)

def save_orientation_final_3d(data: dict[str, np.ndarray], out_dir: Path) -> None:
    valid = data["attitude_valid"] > 0.5
    if not np.any(valid):
        print("No attitude records found. Skipping final orientation plot.")
        return
    idx = np.where(valid)[0][-1]
    body_x, body_y, body_z, nose = get_body_and_nose_vectors(data["q0"][idx], data["q1"][idx], data["q2"][idx], data["q3"][idx])
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    plot_body_and_nose(ax, body_x, body_y, body_z, nose)
    setup_orientation_axes(ax, f"Orientation Only, t = {data['time_s'][idx]:.2f} s")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_dir / "orientation_final_3d.png", dpi=200)
    plt.close(fig)

def save_orientation_video(data: dict[str, np.ndarray], out_dir: Path) -> None:
    valid = data["attitude_valid"] > 0.5
    if not np.any(valid):
        print("No attitude records found. Skipping orientation video.")
        return
    valid_idx = np.where(valid)[0]
    if valid_idx.size > VIDEO_MAX_FRAMES:
        selected = np.linspace(0, valid_idx.size - 1, VIDEO_MAX_FRAMES).astype(int)
        frame_indices = valid_idx[selected]
    else:
        frame_indices = valid_idx
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    def draw_frame(frame_number: int):
        idx = frame_indices[frame_number]
        body_x, body_y, body_z, nose = get_body_and_nose_vectors(data["q0"][idx], data["q1"][idx], data["q2"][idx], data["q3"][idx])
        ax.cla()
        plot_body_and_nose(ax, body_x, body_y, body_z, nose, include_labels=True)
        setup_orientation_axes(ax, f"Orientation Only, t = {data['time_s'][idx]:.2f} s")
        ax.legend(loc="upper right")
        return []
    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / VIDEO_FPS, blit=False)
    mp4_path = out_dir / "orientation_video.mp4"
    gif_path = out_dir / "orientation_video.gif"
    try:
        writer = FFMpegWriter(fps=VIDEO_FPS)
        animation.save(mp4_path, writer=writer, dpi=150)
        print(f"Saved orientation video: {mp4_path}")
    except Exception as e:
        print(f"Could not save MP4, falling back to GIF. Reason: {e}")
        writer = PillowWriter(fps=VIDEO_FPS)
        animation.save(gif_path, writer=writer, dpi=120)
        print(f"Saved orientation GIF: {gif_path}")
    plt.close(fig)

def main() -> None:
    bin_path = SCRIPT_DIR / INPUT_FILE
    if not bin_path.exists():
        raise FileNotFoundError(f"\nCould not find this file:\n{bin_path}\n\nMake sure no_gps.py and {INPUT_FILE} are in the SAME folder,\nor change INPUT_FILE at the top of no_gps.py.")
    if OUTPUT_DIR is None:
        out_dir = SCRIPT_DIR / f"{bin_path.stem}_no_gps_decoded"
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
    csv_path = out_dir / f"{bin_path.stem}_no_gps.csv"
    write_csv(records, csv_path)
    data = rows_to_arrays(records)
    save_summary(data, header, out_dir)
    bad_crc_count = int(np.sum(data["crc_ok"] < 0.5))
    if bad_crc_count:
        print(f"Warning: {bad_crc_count} records failed CRC.")
    if MAKE_PLOTS:
        save_orientation_graphs(data, out_dir)
        save_rocket_nose_graphs(data, out_dir)
        save_orientation_final_3d(data, out_dir)
    if MAKE_VIDEO:
        save_orientation_video(data, out_dir)
    print("")
    print("Done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {out_dir / 'summary.txt'}")
    print(f"Orientation plot: {out_dir / 'orientation_final_3d.png'}")
    if MAKE_VIDEO:
        print(f"Orientation video: {out_dir / 'orientation_video.mp4'}")

if __name__ == "__main__":
    main()
