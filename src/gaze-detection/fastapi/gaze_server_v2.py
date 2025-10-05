"""
Production-Ready Gaze Tracking Server with Authentication & Session Management
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import json
import mediapipe as mp
import sys
from pathlib import Path
import pickle
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent.parent))
from gaze_tracker_v5 import EnhancedGazeTracker
import auth

app = FastAPI(title="Gaze Tracker API v2 - Production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
mp_face_mesh = mp.solutions.face_mesh

# Models
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class GazeTrackerSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tracker = EnhancedGazeTracker()
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.initialized = False

        # Calibration state
        self.calibration_active = False
        self.calibration_points = []
        self.current_point_idx = 0
        self.calibration_samples = []
        self.samples_for_current_point = []
        self.SAMPLES_PER_POINT = 20  # 20 samples for accuracy
        self.calibration_start_time = None
        self.point_start_time = None
        self.POINT_TIMEOUT_SECONDS = 5

        # Frame optimization
        self.frame_count = 0
        self.last_frame_time = time.time()

    def initialize(self, width, height):
        """Initialize tracker with screen dimensions"""
        if not self.initialized:
            self.tracker.set_screen_size(width, height)
            self.initialized = True

            # Try to load saved calibration from session
            saved_cal = auth.get_calibration(self.session_id)
            if saved_cal:
                self._restore_calibration(saved_cal)

    def _restore_calibration(self, calibration_data):
        """Restore calibration from saved session"""
        try:
            self.tracker.model_direction_x = pickle.loads(calibration_data['model_x'])
            self.tracker.model_direction_y = pickle.loads(calibration_data['model_y'])
            self.tracker.model_direction_z = pickle.loads(calibration_data['model_z'])
            self.tracker.is_calibrated = True
            print(f"Restored calibration for session {self.session_id}")
        except Exception as e:
            print(f"Failed to restore calibration: {e}")

    def _save_calibration(self):
        """Save calibration to session"""
        try:
            calibration_data = {
                'model_x': pickle.dumps(self.tracker.model_direction_x),
                'model_y': pickle.dumps(self.tracker.model_direction_y),
                'model_z': pickle.dumps(self.tracker.model_direction_z),
                'timestamp': datetime.now().isoformat()
            }
            auth.save_calibration(self.session_id, calibration_data)
            print(f"Saved calibration for session {self.session_id}")
        except Exception as e:
            print(f"Failed to save calibration: {e}")

    def start_calibration(self, width, height):
        """Start 25-point calibration (5x5 grid)"""
        if not self.initialized:
            self.initialize(width, height)

        # Generate 5x5 grid of calibration points
        self.calibration_points = []
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)

        for row in range(5):
            for col in range(5):
                x = margin_x + col * (width - 2 * margin_x) // 4
                y = margin_y + row * (height - 2 * margin_y) // 4
                self.calibration_points.append((x, y))

        self.calibration_active = True
        self.current_point_idx = 0
        self.calibration_samples = []
        self.samples_for_current_point = []
        self.calibration_start_time = time.time()
        self.point_start_time = time.time()

        return self.calibration_points[0]

    def process_calibration_frame(self, frame_data):
        """Process frame during calibration with validation and timeout"""
        try:
            # Check point timeout - force move to next point
            if time.time() - self.point_start_time > self.POINT_TIMEOUT_SECONDS:
                print(f"Point {self.current_point_idx + 1} timed out - forcing move to next")

                # Save whatever samples we have (may be incomplete)
                if len(self.samples_for_current_point) > 0:
                    cal_point = self.calibration_points[self.current_point_idx]
                    self.calibration_samples.append({
                        'point': cal_point,
                        'samples': self.samples_for_current_point.copy()
                    })

                self.current_point_idx += 1
                self.samples_for_current_point = []
                self.point_start_time = time.time()

                if self.current_point_idx >= len(self.calibration_points):
                    # Calibration complete (with timeouts)
                    success = self._train_calibration()
                    self.calibration_active = False

                    if success:
                        self._save_calibration()
                        return {
                            'status': 'complete',
                            'message': 'Calibration complete (some points timed out)',
                            'total_time': round(time.time() - self.calibration_start_time, 1)
                        }
                    else:
                        return {
                            'status': 'error',
                            'message': 'Calibration failed - please try again'
                        }
                else:
                    # Move to next point
                    next_point = self.calibration_points[self.current_point_idx]
                    return {
                        'status': 'next_point',
                        'point_index': self.current_point_idx,
                        'point': next_point,
                        'total_points': len(self.calibration_points),
                        'message': 'Previous point timed out'
                    }

            img_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {'status': 'error', 'message': 'Invalid frame'}

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                return {'status': 'no_face', 'message': 'No face detected - please face the camera'}

            face_landmarks = results.multi_face_landmarks[0]

            # Collect samples for current point (removed visibility check - MediaPipe doesn't provide it reliably)
            if len(self.samples_for_current_point) < self.SAMPLES_PER_POINT:
                self.samples_for_current_point.append((face_landmarks, w, h))

                progress = len(self.samples_for_current_point) / self.SAMPLES_PER_POINT

                return {
                    'status': 'collecting',
                    'point_index': self.current_point_idx,
                    'progress': progress,
                    'samples_collected': len(self.samples_for_current_point),
                    'total_points': len(self.calibration_points)
                }

            # Point complete - save and move to next
            cal_point = self.calibration_points[self.current_point_idx]
            self.calibration_samples.append({
                'point': cal_point,
                'samples': self.samples_for_current_point.copy()
            })

            print(f"Point {self.current_point_idx + 1}/{len(self.calibration_points)} complete - collected {len(self.samples_for_current_point)} samples")

            self.current_point_idx += 1
            self.samples_for_current_point = []
            self.point_start_time = time.time()  # Reset timeout

            if self.current_point_idx >= len(self.calibration_points):
                # Calibration complete, train model
                print("All points collected, training model...")
                success = self._train_calibration()
                self.calibration_active = False

                if success:
                    self._save_calibration()
                    print("Calibration complete and saved!")
                    return {
                        'status': 'complete',
                        'message': 'Calibration complete! You can now use gaze tracking.',
                        'total_time': round(time.time() - self.calibration_start_time, 1)
                    }
                else:
                    print("Calibration training failed!")
                    return {
                        'status': 'error',
                        'message': 'Calibration failed - please try again'
                    }
            else:
                # Move to next point
                next_point = self.calibration_points[self.current_point_idx]
                print(f"Moving to point {self.current_point_idx + 1}/{len(self.calibration_points)}: {next_point}")
                return {
                    'status': 'next_point',
                    'point_index': self.current_point_idx,
                    'point': next_point,
                    'total_points': len(self.calibration_points)
                }

        except Exception as e:
            print(f"Calibration error: {e}")
            return {'status': 'error', 'message': str(e)}

    def _train_calibration(self):
        """Train the gaze model using collected calibration samples"""
        try:
            # Use the tracker's calibration method
            for cal_data in self.calibration_samples:
                cal_point = cal_data['point']
                samples = cal_data['samples']

                for face_landmarks, w, h in samples:
                    self.tracker.add_calibration_point(face_landmarks, w, h, cal_point)

            # Train the models using calibrate() method
            success = self.tracker.calibrate()

            if success:
                print(f"Calibration training complete for session {self.session_id}")
                return True
            else:
                print(f"Calibration failed - not enough samples")
                return False

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_frame(self, frame_data):
        """Process base64 encoded frame and return gaze coordinates with optimization"""
        try:
            # Frame skipping optimization (process every 2nd frame)
            self.frame_count += 1
            if self.frame_count % 2 == 0:
                return None  # Skip this frame

            current_time = time.time()
            fps = 1.0 / (current_time - self.last_frame_time) if (current_time - self.last_frame_time) > 0 else 0
            self.last_frame_time = current_time

            img_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            h, w = frame.shape[:2]

            if not self.initialized:
                self.initialize(w, h)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                if self.tracker.is_calibrated:
                    gaze_point, confidence_x, confidence_y = self.tracker.predict_gaze(
                        face_landmarks, w, h
                    )

                    if gaze_point is not None:
                        return {
                            'x': float(gaze_point[0]),
                            'y': float(gaze_point[1]),
                            'confidence_x': float(confidence_x),
                            'confidence_y': float(confidence_y),
                            'calibrated': True,
                            'fps': round(fps, 1),
                            'timestamp': current_time
                        }

                return {
                    'x': None,
                    'y': None,
                    'confidence_x': 0.0,
                    'confidence_y': 0.0,
                    'calibrated': self.tracker.is_calibrated,
                    'fps': round(fps, 1)
                }

            return None

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Active sessions
active_sessions = {}

# API Endpoints
@app.post("/register")
async def register(user: UserRegister):
    """Register a new user"""
    created_user = auth.create_user(user.username, user.password)
    if not created_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User created successfully", "username": user.username}

@app.post("/login")
async def login(user: UserLogin):
    """Login and get access token"""
    authenticated_user = auth.authenticate_user(user.username, user.password)
    if not authenticated_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = auth.create_access_token(data={"sub": user.username})
    session_id = auth.create_session(user.username)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "session_id": session_id,
        "username": user.username
    }

@app.websocket("/ws/gaze/{session_id}")
async def websocket_gaze_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time gaze tracking with session management"""
    await websocket.accept()

    # Verify session
    session_data = auth.get_session(session_id)
    if not session_data:
        await websocket.send_json({
            'type': 'error',
            'data': {'message': 'Invalid session'}
        })
        await websocket.close()
        return

    # Create or get tracker session
    if session_id not in active_sessions:
        active_sessions[session_id] = GazeTrackerSession(session_id)

    session = active_sessions[session_id]

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get('type') == 'init':
                width = message.get('width', 1920)
                height = message.get('height', 1080)
                session.initialize(width, height)

                await websocket.send_json({
                    'type': 'initialized',
                    'data': {
                        'calibrated': session.tracker.is_calibrated,
                        'session_id': session_id
                    }
                })

            elif message.get('type') == 'frame':
                frame_data = message.get('data')
                result = session.process_frame(frame_data)

                if result:
                    await websocket.send_json({
                        'type': 'gaze',
                        'data': result
                    })

            elif message.get('type') == 'start_calibration':
                width = message.get('width', 1920)
                height = message.get('height', 1080)

                first_point = session.start_calibration(width, height)

                await websocket.send_json({
                    'type': 'calibration_started',
                    'data': {
                        'point_index': 0,
                        'point': first_point,
                        'total_points': 25,
                        'message': 'Look at the red dot and hold still'
                    }
                })

            elif message.get('type') == 'calibration_frame':
                if not session.calibration_active:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'message': 'Calibration not active'}
                    })
                    continue

                frame_data = message.get('data')
                result = session.process_calibration_frame(frame_data)

                await websocket.send_json({
                    'type': 'calibration_update',
                    'data': result
                })

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Don't cleanup session - keep it for reconnection
        pass

@app.get("/")
async def root():
    try:
        with open(Path(__file__).parent / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading page: {str(e)}</h1>", status_code=500)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "total_users": len(auth.users_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
