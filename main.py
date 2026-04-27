import cv2
import yaml
import os
import pandas as pd
import json
import time
import webbrowser
import threading
import numpy as np
from datetime import datetime
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# Revert to standard TCP for stability (some cameras crash with nobuffer)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import traceback

def load_config(config_path="config/settings.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_break_info():
    now = datetime.now()
    day = now.weekday() # 0=Mon, 4=Fri
    hour = now.hour
    minute = now.minute
    
    # 9:00 - 9:15 Team Break
    if hour == 9 and minute < 15:
        return "TEA BREAK IN PROGRESS"
    
    # 1:00 PM Lunch Break
    if hour == 13:
        if day == 4: # Friday: 1:00 - 2:00
            return "FRIDAY PRAYER / LUNCH BREAK"
        elif minute < 30: # Other days: 1:00 - 1:30
            return "LUNCH BREAK IN PROGRESS"
            
    # 3:00 - 3:20 Tea Break (Extended buffer for demo stability)
    if hour == 15 and minute < 20:
        return "TEA BREAK IN PROGRESS"
        
    return None

def start_server(port=8000):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args): return
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            super().end_headers()
    while port < 8010:
        try:
            with TCPServer(("", port), QuietHandler) as httpd:
                with open("logs/current_port.txt", "w") as f: f.write(str(port))
                httpd.serve_forever()
        except OSError: port += 1

def main():
    config = load_config()
    employee_df = pd.read_csv("data/employees.csv")
    employee_map = {int(r['marker_id']): {"name": r['name'], "dept": r.get('department', 'Production')} for _, r in employee_df.iterrows()}
    
    model = YOLO(config['detection']['model_path'])
    enabled_cameras = [c for c in config['cameras'] if c['enabled']]
    caps = []
    for cam in enabled_cameras:
        cap = cv2.VideoCapture(cam['video_path'], cv2.CAP_FFMPEG)
        if cap.isOpened():
            caps.append({"id": cam['id'], "cap": cap, "url": cam['video_path'], "last_frame": None, "err_count": 0})

    if not caps: return

    history = {} 
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs/live", exist_ok=True)
    os.makedirs("outputs/paths", exist_ok=True)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config['aruco']['dictionary_type']))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(1)
    
    session_start = time.time()
    scale = config['mapping'].get('scale_factor', 50.0)

    try:
        while True:
            try:
                now_ts = time.time()
                break_msg = get_break_info()
                
                frame_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "uptime": int(now_ts - session_start),
                    "cameras": [],
                    "total_person_count": 0,
                    "stats": [],
                    "lunch_mode": break_msg is not None,
                    "break_message": break_msg or ""
                }
                
                total_detected_people = 0
                
                for cam in caps:
                    ret, frame = cam['cap'].read()
                    if not ret or frame is None or np.mean(frame) < 5 or np.mean(frame) > 250:
                        cam['err_count'] += 1
                        if cam['err_count'] > 5:
                            cam['cap'].release()
                            cam['cap'] = cv2.VideoCapture(cam['url'], cv2.CAP_FFMPEG)
                            cam['err_count'] = 0
                        continue
                    
                    # Performance: Resize frame for faster AI/ArUco detection
                    render_frame = frame.copy()
                    proc_w = 1024
                    h, w = frame.shape[:2]
                    scale_r = w / proc_w
                    frame = cv2.resize(frame, (proc_w, int(h * (proc_w/w))))
                    
                    cam['last_frame'] = render_frame 
                    
                    # Always track — lunch_mode only affects the header tag on dashboard
                    results = model(frame, conf=0.25, classes=[0], verbose=False)[0]
                    total_detected_people += len(results.boxes)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = detector.detectMarkers(gray)

                    if ids is not None:
                        for i, marker_id in enumerate(ids.flatten()):
                            marker_id = int(marker_id)
                            emp = employee_map.get(marker_id)
                            if not emp: continue
                            
                            # Scale coordinate back to original frame size for consistent measurement
                            raw_pos = corners[i][0].mean(axis=0) * scale_r
                            
                            if marker_id not in history:
                                history[marker_id] = {
                                    "name": emp['name'], "dept": emp['dept'],
                                    "pos_buffer": deque(maxlen=5), 
                                    "path": [raw_pos.tolist()],
                                    "dist": 0.0, "status": "WORKING", "last_move_ts": now_ts,
                                    "cam_id": cam['id']
                                }
                            
                            h = history[marker_id]
                            h["pos_buffer"].append(raw_pos)
                            smooth_pos = np.mean(h["pos_buffer"], axis=0)
                            last_pt = np.array(h["path"][-1])
                            d_pixels = float(np.sqrt(np.sum((smooth_pos - last_pt)**2)))
                            
                            # Lower threshold (8px) for better sensitivity
                            if d_pixels > 8:
                                dist_meters = d_pixels / scale
                                dist_feet = dist_meters * 3.28084
                                
                                # JUMP FILTER: If worker 'teleports' > 20ft in one frame, ignore as noise
                                if dist_feet > 20 and len(h["path"]) > 1:
                                    h["pos_buffer"].clear() # Reset buffer on jump
                                    continue

                                h["dist"] += dist_feet
                                h["path"].append(smooth_pos.tolist())
                                h["status"] = "WALKING"
                                h["last_move_ts"] = now_ts
                            elif now_ts - h["last_move_ts"] > 5:
                                h["status"] = "WORKING"

                            pts = (corners[i][0] * scale_r).astype(int)
                            cv2.polylines(render_frame, [pts], True, (34, 197, 94), 2)
                            cv2.putText(render_frame, emp['name'], (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Atomic save to prevent black/grey frames during dashboard refresh
                    live_path = f"outputs/live/{cam['id']}.jpg"
                    temp_path = f"outputs/live/{cam['id']}_tmp.jpg"
                    cv2.imwrite(temp_path, render_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    try:
                        os.replace(temp_path, live_path) # Atomic swap
                    except:
                        pass
                    
                    frame_data["cameras"].append({"id": cam['id'], "person_count": len(results.boxes), "live_view": live_path})


                for mid, s in history.items():
                        target_cam = next((c for c in caps if c['id'] == s['cam_id']), None)
                        if target_cam and target_cam['last_frame'] is not None:
                            p_img = target_cam['last_frame'].copy()
                            pts = np.array(s['path'], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(p_img, [pts], False, (0, 255, 255), 3)
                            if len(pts) > 0: cv2.circle(p_img, tuple(pts[-1][0]), 8, (0, 0, 255), -1)
                            cv2.imwrite(f"outputs/paths/{mid}.jpg", p_img, [cv2.IMWRITE_JPEG_QUALITY, 50])

                        frame_data["stats"].append({
                            "id": mid, "name": s["name"], "dept": s["dept"],
                            "dist": float(round(s["dist"], 2)), "status": s["status"],
                            "path_view": f"/outputs/paths/{mid}.jpg"
                        })
                
                with open("logs/live_stats.json", 'w') as f: json.dump(frame_data, f, indent=2)
                time.sleep(0.01)

            except Exception as e:
                print(f"INNER LOOP ERROR: {e}")
                traceback.print_exc()
                time.sleep(1)
                
    except KeyboardInterrupt: pass
    finally:
        for cam in caps: cam['cap'].release()

if __name__ == "__main__":
    main()
