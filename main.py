import time
import cv2
import numpy as np
import torch
import json
import threading
import asyncio
import sqlite3
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

# 데이터베이스 초기화 (존 관련 사이클 타임 기록 제외, 충돌 알람 및 객체 로그만 유지)
def init_db():
    conn = sqlite3.connect("bev_data.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            class_name TEXT,
            map_x REAL,
            map_y REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients = set()
latest_frame = None
latest_data = {"alerts": []}
db_conn = init_db()

def calculate_distance(x1, y1, x2, y2):
    raw_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    real_dist = raw_dist * (0.26 / 1.2)
    return real_dist

def run_ai_loop():
    global latest_frame, latest_data
    
    try:
        cameraMatrix = np.load("cameraMatrix.npy")
        distCoeffs = np.load("distCoeffs.npy")
        H_bev_to_map = np.load("H_bev_to_map.npy")
    except FileNotFoundError:
        print("경고: npy 파일이 존재하지 않음.")
        return
    
    MODEL_PATH = "/home/chansik/Downloads/car_b_yolov8n_result/content/runs/detect/car_b_yolov8n_b16_p80/weights/best.pt"
    model = YOLO(MODEL_PATH)
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(2)
    src = np.float32([[63, 103], [609, 85], [622, 357], [69, 376]])
    WIDTH, HEIGHT = 600, 400
    dst = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    
    def bev_to_map(u, v):
        p = np.array([[[u, v]]], dtype=np.float32)
        out = cv2.perspectiveTransform(p, H_bev_to_map)
        return float(out[0][0][0]), float(out[0][0][1])

    cursor = db_conn.cursor()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs)
        bev = cv2.warpPerspective(undistorted, matrix, (WIDTH, HEIGHT))
        
        # half 파라미터 삭제 처리 (경고 문구 제거)
        results = model.track(source=bev, device=DEVICE, conf=0.5, persist=True, imgsz=640, verbose=False)
        
        frame_data = {"alerts": []}
        objects_info = []
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                track_id = int(box.id[0]) if box.id is not None else -1
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                map_x, map_y = bev_to_map(cx, cy)
                
                obj_dict = {"id": track_id, "class": label, "map_x": map_x, "map_y": map_y, "px": cx, "py": cy}
                objects_info.append(obj_dict)

                cursor.execute("INSERT INTO object_logs (track_id, class_name, map_x, map_y) VALUES (?, ?, ?, ?)", 
                               (track_id, label, map_x, map_y))
                
                # 객체 위치 및 식별자 렌더링 (OpenCV)
                color = (0, 0, 255) if "car" in label.lower() else (0, 255, 255)
                cv2.circle(bev, (cx, cy), 8, color, -1)
                cv2.putText(bev, f"{label}[{track_id}]", (cx + 12, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 충돌 위험 판정 및 렌더링 (OpenCV)
        y_offset = 30
        for i in range(len(objects_info)):
            for j in range(i + 1, len(objects_info)):
                obj1 = objects_info[i]
                obj2 = objects_info[j]
                dist = calculate_distance(obj1["map_x"], obj1["map_y"], obj2["map_x"], obj2["map_y"])
                
                if dist < 0.3:
                    msg = f"충돌 위험! {obj1['class']}[{obj1['id']}] - {obj2['class']}[{obj2['id']}] 거리: {dist:.2f}m"
                    frame_data["alerts"].append({"msg": msg})
                    cursor.execute("INSERT INTO event_logs (event_type, message) VALUES (?, ?)", ("COLLISION_WARNING", msg))
                    
                    cv2.line(bev, (obj1["px"], obj1["py"]), (obj2["px"], obj2["py"]), (0, 0, 255), 4)
                    cv2.putText(bev, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25
        
        db_conn.commit()
        latest_data = frame_data
        
        # JPEG 품질 70 압축 적용
        ret_jpg, buffer = cv2.imencode('.jpg', bev, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret_jpg:
            latest_frame = buffer.tobytes()

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=run_ai_loop, daemon=True)
    t.start()

def generate_video():
    while True:
        if latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.1) # 10 FPS 전송 제한 적용

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")

async def send_ws_data():
    while True:
        if connected_clients and latest_data:
            msg = json.dumps(latest_data)
            for client in list(connected_clients):
                try:
                    await client.send_text(msg)
                except:
                    connected_clients.remove(client)
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def start_ws_sender():
    asyncio.create_task(send_ws_data())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)