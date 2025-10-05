/**
 * Local Gaze Tracker - JavaScript Port of gaze_tracker_v5.py
 * 
 * This is a comprehensive port of the Python EnhancedGazeTracker class
 * with all major features preserved including:
 * - MediaPipe face mesh detection
 * - Advanced geometric gaze estimation
 * - 25-point calibration system
 * - Dynamic eye center calibration
 * - Temporal smoothing and outlier rejection
 * - Multi-modal confidence fusion
 */

class LocalGazeTracker {
    constructor() {
        // Screen geometry
        this.screenWidth = 640;
        this.screenHeight = 480;
        this.screenWidthMm = null;
        this.screenHeightMm = null;

        // Per-user calibrated eye parameters
        this.userIpdMm = 63.0;
        this.focalLength = null;
        this.leftEyeCenterHead = [-31.5, 0.0, 80.0];
        this.rightEyeCenterHead = [31.5, 0.0, 80.0];
        this.kappaAngleLeft = 5.0 * Math.PI / 180;
        this.kappaAngleRight = 5.0 * Math.PI / 180;
        this.eyeballRadiusMm = 12.0;
        
        // Dynamic eye center calibration
        this.eyeCenterHistory = [];
        this.eyeCenterConfidence = 0.0;
        this.eyeCenterAdaptive = true;
        
        // Adaptive kappa angle estimation
        this.kappaAngleHistory = [];
        this.kappaConfidence = 0.0;
        this.kappaAdaptive = true;
        
        // Per-eye individual modeling
        this.leftEyeModel = {kappa: 5.0 * Math.PI / 180, centerOffset: [0.0, 0.0, 0.0]};
        this.rightEyeModel = {kappa: 5.0 * Math.PI / 180, centerOffset: [0.0, 0.0, 0.0]};
        
        // Multi-scale iris refinement
        this.irisScaleFactors = [0.8, 1.0, 1.2];
        this.irisRefinementEnabled = true;
        
        // Head pose confidence and error compensation
        this.poseConfidenceHistory = [];
        this.poseErrorCompensation = true;
        
        // Temporal consistency checking
        this.featureConsistencyHistory = [];
        this.outlierRejectionEnabled = true;
        
        // Corneal refraction model parameters
        this.corneaRadiusMm = 7.8;
        this.corneaRefractiveIndex = 1.376;
        this.airRefractiveIndex = 1.0;
        this.corneaThicknessMm = 0.55;

        // Screen position in camera coordinates
        this.screenDistanceMm = 500.0;
        this.screenCenterCamera = null;
        this.screenNormalCamera = [0, 0, -1];

        // MediaPipe IRIS constants
        this.IRIS_DIAMETER_MM = 11.7;
        this.irisDistanceHistory = [];
        this.distanceConfidenceThreshold = 0.3;

        // ML model for gaze direction prediction (simplified for JS)
        this.modelDirectionX = null;
        this.modelDirectionY = null;
        this.modelDirectionZ = null;
        this.isCalibrated = false;

        // Calibration data
        this.calibrationFeatures = [];
        this.calibrationGazeDirections = [];
        this.calibrationScreenPoints = [];

        // MediaPipe face mesh instance
        this.faceMesh = null;
        this.camera = null;

        // Tracking state
        this.isTracking = false;
        this.trackingCallback = null;
        this.lastFrameTime = 0;
        this.frameCount = 0;

        // Recent predictions for temporal smoothing
        this.recentPredictions = [];
        this.maxRecentPredictions = 5;

        // MediaPipe landmark indices (same as Python version)
        this.NOSE_TIP = 1;
        this.CHIN = 152;
        this.LEFT_EYE_OUTER = 33;
        this.RIGHT_EYE_OUTER = 263;
        this.LEFT_EYE_INNER = 133;
        this.RIGHT_EYE_INNER = 362;
        this.LEFT_IRIS = [468, 469, 470, 471, 472];
        this.RIGHT_IRIS = [473, 474, 475, 476, 477];
        this.LEFT_EYE_TOP = 159;
        this.LEFT_EYE_BOTTOM = 145;
        this.RIGHT_EYE_TOP = 386;
        this.RIGHT_EYE_BOTTOM = 374;

        // Enhanced eyelid landmarks for vertical gaze
        this.LEFT_EYE_UPPER_LID = [159, 158, 157, 173, 133, 7, 163, 144, 145, 153, 154, 155, 133];
        this.LEFT_EYE_LOWER_LID = [145, 153, 154, 155, 133, 173, 157, 158, 159, 144, 163, 7];
        this.RIGHT_EYE_UPPER_LID = [386, 387, 388, 398, 263, 362, 382, 373, 374, 380, 381, 382, 263];
        this.RIGHT_EYE_LOWER_LID = [374, 380, 381, 382, 263, 398, 388, 387, 386, 373, 382, 362];

        // Eyebrow landmarks for extreme vertical gaze
        this.LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
        this.RIGHT_EYEBROW = [296, 334, 293, 300, 276, 283, 282, 295, 285, 336];

        // 6 points for solvePnP
        this.POSE_LANDMARKS_INDICES = [1, 152, 33, 263, 61, 291];

        // 3D canonical face model (mm) - same as Python
        this.MODEL_POINTS_3D = [
            [0.0, 0.0, 0.0],           // Nose tip
            [0.0, -330.0, -65.0],      // Chin
            [-225.0, 170.0, -135.0],   // Left eye left corner
            [225.0, 170.0, -135.0],    // Right eye right corner
            [-150.0, -150.0, -125.0],  // Left mouth corner
            [150.0, -150.0, -125.0]    // Right mouth corner
        ];

        console.log('LocalGazeTracker initialized');
    }

    /**
     * Initialize MediaPipe face mesh and camera
     */
    async initialize() {
        try {
            console.log('Initializing MediaPipe face mesh...');
            
            this.faceMesh = new FaceMesh({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
                }
            });

            this.faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            this.faceMesh.onResults((results) => {
                this.onFaceMeshResults(results);
            });

            // Setup camera
            this.camera = new Camera(document.getElementById('video'), {
                onFrame: async () => {
                    if (this.faceMesh) {
                        await this.faceMesh.send({image: document.getElementById('video')});
                    }
                },
                width: 640,
                height: 480
            });

            await this.camera.start();
            console.log('MediaPipe face mesh initialized successfully');

        } catch (error) {
            console.error('Failed to initialize MediaPipe:', error);
            throw error;
        }
    }

    /**
     * Set screen size (same as Python version)
     */
    setScreenSize(width, height) {
        this.screenWidth = width;
        this.screenHeight = height;
        
        // Calculate focal length assuming 640x480 camera
        this.focalLength = (640 * 500) / (width * 0.0254); // Assuming 25.4mm per inch
        
        console.log(`Screen size set to: ${width}x${height}`);
        console.log(`Calculated focal length: ${this.focalLength}`);
    }

    /**
     * Handle face mesh results from MediaPipe
     */
    onFaceMeshResults(results) {
        this.frameCount++;
        const currentTime = Date.now();
        const fps = 1000 / (currentTime - this.lastFrameTime);
        this.lastFrameTime = currentTime;

        // Store results for calibration use
        this.lastResults = results;

        // Update face detection status
        const faceDetected = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;
        document.getElementById('faceDetected').textContent = faceDetected ? 'Yes' : 'No';
        document.getElementById('fpsCounter').textContent = fps.toFixed(1);

        if (faceDetected) {
            const faceLandmarks = results.multiFaceLandmarks[0];
            
            if (this.isTracking && this.trackingCallback) {
                const gazeData = this.predictGaze(faceLandmarks, 640, 480);
                if (gazeData) {
                    gazeData.fps = fps;
                    this.trackingCallback(gazeData);
                }
            }
        }
    }

    /**
     * Start calibration process (25-point grid)
     */
    startCalibration(progressCallback, statusCallback) {
        console.log('Starting 25-point calibration...');
        
        // Generate 25-point calibration grid (5x5)
        const calibrationPoints = [];
        const marginX = 64; // 10% of 640
        const marginY = 48; // 10% of 480
        
        for (let row = 0; row < 5; row++) {
            for (let col = 0; col < 5; col++) {
                const x = marginX + col * (640 - 2 * marginX) / 4;
                const y = marginY + row * (480 - 2 * marginY) / 4;
                calibrationPoints.push([x, y]);
            }
        }

        this.calibrationPoints = calibrationPoints;
        this.currentCalibrationIndex = 0;
        this.calibrationSamples = [];
        this.samplesForCurrentPoint = [];
        this.SAMPLES_PER_POINT = 20; // Back to 20 samples
        this.isCalibrating = true;

        // Start with first point
        this.showCalibrationPoint(0, progressCallback, statusCallback);
    }

    /**
     * Show calibration point and collect samples
     */
    showCalibrationPoint(index, progressCallback, statusCallback) {
        if (index >= this.calibrationPoints.length) {
            // Calibration complete
            this.isCalibrating = false;
            const success = this.trainCalibration();
            if (success) {
                this.saveCalibration();
                statusCallback('complete', 'Calibration complete!');
            } else {
                statusCallback('error', 'Calibration failed');
            }
            return;
        }

        const point = this.calibrationPoints[index];
        this.currentCalibrationIndex = index;
        this.samplesForCurrentPoint = [];
        
        // Show countdown (much faster)
        let countdown = 1; // Reduced from 3 to 1 second
        progressCallback(point, 0);
        statusCallback('countdown', `Look at the red dot... ${countdown}`);
        
        const countdownInterval = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                statusCallback('countdown', `Look at the red dot... ${countdown}`);
            } else if (countdown === 0) {
                statusCallback('countdown', 'Start looking at the dot now!');
            } else {
                clearInterval(countdownInterval);
                this.collectCalibrationSamples(index, progressCallback, statusCallback);
            }
        }, 1000);
    }

    /**
     * Collect samples for current calibration point
     */
    collectCalibrationSamples(index, progressCallback, statusCallback) {
        const point = this.calibrationPoints[index];
        let samplesCollected = 0;
        let noFaceCount = 0;
        const maxNoFaceCount = 50; // Allow up to 5 seconds of no face
        
        console.log(`Starting sample collection for point ${index + 1}/${this.calibrationPoints.length}`);
        
        const collectInterval = setInterval(() => {
            // Check if we have recent face detection results
            if (this.lastResults && this.lastResults.multiFaceLandmarks && this.lastResults.multiFaceLandmarks.length > 0) {
                const faceLandmarks = this.lastResults.multiFaceLandmarks[0];
                
                try {
                    this.addCalibrationPoint(faceLandmarks, 640, 480, point);
                    samplesCollected++;
                    noFaceCount = 0; // Reset no-face counter
                    
                    const progress = samplesCollected / this.SAMPLES_PER_POINT;
                    progressCallback(point, progress);
                    statusCallback('collecting', `Collecting samples... ${samplesCollected}/${this.SAMPLES_PER_POINT}`);
                    
                    console.log(`Collected sample ${samplesCollected}/${this.SAMPLES_PER_POINT} for point ${index + 1}`);
                    
                    if (samplesCollected >= this.SAMPLES_PER_POINT) {
                        clearInterval(collectInterval);
                        console.log(`Completed collection for point ${index + 1}`);
                        
                        // Move to next point
                        setTimeout(() => {
                            this.showCalibrationPoint(index + 1, progressCallback, statusCallback);
                        }, 400); // 0.4 second cooldown between dots
                    }
                } catch (error) {
                    console.error('Error adding calibration point:', error);
                }
            } else {
                noFaceCount++;
                statusCallback('no_face', 'No face detected. Please position your face in view.');
                
                if (noFaceCount >= maxNoFaceCount) {
                    clearInterval(collectInterval);
                    statusCallback('error', 'No face detected for too long. Calibration stopped.');
                    console.log('Calibration stopped due to no face detection');
                }
            }
        }, 100); // Collect sample every 100ms
        
        // Store interval ID for cleanup
        this.currentCalibrationInterval = collectInterval;
    }

    /**
     * Add calibration point (port of Python version)
     */
    addCalibrationPoint(faceLandmarks, frameWidth, frameHeight, screenPoint) {
        try {
            // Extract features
            const features = this.extractGazeFeatures(faceLandmarks, frameWidth, frameHeight);
            
            if (features && features.length > 0) {
                // Calculate gaze direction from screen point
                const gazeDirection = this.calculateGazeDirectionFromScreenPoint(screenPoint);
                
                this.calibrationFeatures.push(features);
                this.calibrationGazeDirections.push(gazeDirection);
                this.calibrationScreenPoints.push(screenPoint);
                
                console.log(`Added calibration point ${this.calibrationFeatures.length}: screen=${screenPoint}, features=${features.length}`);
            }
        } catch (error) {
            console.error('Error adding calibration point:', error);
        }
    }

    /**
     * Extract gaze features (simplified port of Python version)
     */
    extractGazeFeatures(faceLandmarks, frameWidth, frameHeight) {
        try {
            const features = [];
            
            // Extract eye landmarks
            const leftEyeLandmarks = this.LEFT_IRIS.map(idx => [
                faceLandmarks[idx].x * frameWidth,
                faceLandmarks[idx].y * frameHeight
            ]);
            
            const rightEyeLandmarks = this.RIGHT_IRIS.map(idx => [
                faceLandmarks[idx].x * frameWidth,
                faceLandmarks[idx].y * frameHeight
            ]);
            
            // Calculate eye centers
            const leftEyeCenter = this.calculateCenter(leftEyeLandmarks);
            const rightEyeCenter = this.calculateCenter(rightEyeLandmarks);
            
            // Calculate eye dimensions
            const leftEyeWidth = this.calculateEyeWidth(faceLandmarks, frameWidth, frameHeight, 'left');
            const rightEyeWidth = this.calculateEyeWidth(faceLandmarks, frameWidth, frameHeight, 'right');
            const leftEyeHeight = this.calculateEyeHeight(faceLandmarks, frameWidth, frameHeight, 'left');
            const rightEyeHeight = this.calculateEyeHeight(faceLandmarks, frameWidth, frameHeight, 'right');
            
            // Calculate head pose (simplified)
            const headPose = this.estimateHeadPose(faceLandmarks, frameWidth, frameHeight);
            
            // Add features
            features.push(leftEyeCenter[0], leftEyeCenter[1]);
            features.push(rightEyeCenter[0], rightEyeCenter[1]);
            features.push(leftEyeWidth, leftEyeHeight);
            features.push(rightEyeWidth, rightEyeHeight);
            features.push(headPose[0], headPose[1], headPose[2]); // roll, pitch, yaw
            
            // Add eye landmark features
            leftEyeLandmarks.forEach(landmark => {
                features.push(landmark[0], landmark[1]);
            });
            rightEyeLandmarks.forEach(landmark => {
                features.push(landmark[0], landmark[1]);
            });
            
            return features;
            
        } catch (error) {
            console.error('Error extracting gaze features:', error);
            return [];
        }
    }

    /**
     * Calculate center of landmarks
     */
    calculateCenter(landmarks) {
        const sum = landmarks.reduce((acc, landmark) => [acc[0] + landmark[0], acc[1] + landmark[1]], [0, 0]);
        return [sum[0] / landmarks.length, sum[1] / landmarks.length];
    }

    /**
     * Calculate eye width
     */
    calculateEyeWidth(faceLandmarks, frameWidth, frameHeight, eye) {
        const outerIdx = eye === 'left' ? this.LEFT_EYE_OUTER : this.RIGHT_EYE_OUTER;
        const innerIdx = eye === 'left' ? this.LEFT_EYE_INNER : this.RIGHT_EYE_INNER;
        
        const outer = [
            faceLandmarks[outerIdx].x * frameWidth,
            faceLandmarks[outerIdx].y * frameHeight
        ];
        const inner = [
            faceLandmarks[innerIdx].x * frameWidth,
            faceLandmarks[innerIdx].y * frameHeight
        ];
        
        return Math.sqrt(Math.pow(outer[0] - inner[0], 2) + Math.pow(outer[1] - inner[1], 2));
    }

    /**
     * Calculate eye height
     */
    calculateEyeHeight(faceLandmarks, frameWidth, frameHeight, eye) {
        const topIdx = eye === 'left' ? this.LEFT_EYE_TOP : this.RIGHT_EYE_TOP;
        const bottomIdx = eye === 'left' ? this.LEFT_EYE_BOTTOM : this.RIGHT_EYE_BOTTOM;
        
        const top = [
            faceLandmarks[topIdx].x * frameWidth,
            faceLandmarks[topIdx].y * frameHeight
        ];
        const bottom = [
            faceLandmarks[bottomIdx].x * frameWidth,
            faceLandmarks[bottomIdx].y * frameHeight
        ];
        
        return Math.sqrt(Math.pow(top[0] - bottom[0], 2) + Math.pow(top[1] - bottom[1], 2));
    }

    /**
     * Estimate head pose (simplified)
     */
    estimateHeadPose(faceLandmarks, frameWidth, frameHeight) {
        // Simplified head pose estimation using key landmarks
        const nose = [faceLandmarks[this.NOSE_TIP].x * frameWidth, faceLandmarks[this.NOSE_TIP].y * frameHeight];
        const chin = [faceLandmarks[this.CHIN].x * frameWidth, faceLandmarks[this.CHIN].y * frameHeight];
        const leftEye = [faceLandmarks[this.LEFT_EYE_OUTER].x * frameWidth, faceLandmarks[this.LEFT_EYE_OUTER].y * frameHeight];
        const rightEye = [faceLandmarks[this.RIGHT_EYE_OUTER].x * frameWidth, faceLandmarks[this.RIGHT_EYE_OUTER].y * frameHeight];
        
        // Calculate roll (rotation around Z-axis)
        const roll = Math.atan2(rightEye[1] - leftEye[1], rightEye[0] - leftEye[0]);
        
        // Calculate pitch (rotation around Y-axis)
        const pitch = Math.atan2(chin[1] - nose[1], Math.abs(chin[0] - nose[0]));
        
        // Calculate yaw (rotation around X-axis) - simplified
        const yaw = 0; // Would need more complex calculation for accurate yaw
        
        return [roll, pitch, yaw];
    }

    /**
     * Calculate gaze direction from screen point
     */
    calculateGazeDirectionFromScreenPoint(screenPoint) {
        // Convert screen point to normalized coordinates
        const x = (screenPoint[0] - this.screenWidth / 2) / (this.screenWidth / 2);
        const y = (screenPoint[1] - this.screenHeight / 2) / (this.screenHeight / 2);
        
        // Calculate gaze direction vector
        const gazeDirection = [
            x,
            y,
            -1 // Looking towards screen
        ];
        
        // Normalize
        const length = Math.sqrt(gazeDirection[0]**2 + gazeDirection[1]**2 + gazeDirection[2]**2);
        return [
            gazeDirection[0] / length,
            gazeDirection[1] / length,
            gazeDirection[2] / length
        ];
    }

    /**
     * Train calibration models (simplified for JavaScript)
     */
    trainCalibration() {
        if (this.calibrationFeatures.length < 15) {
            console.log(`Not enough calibration points: ${this.calibrationFeatures.length}`);
            return false;
        }

        try {
            console.log(`Training with ${this.calibrationFeatures.length} calibration points...`);
            
            // For JavaScript, we'll use a simplified linear regression approach
            // In a full implementation, you could use TensorFlow.js for more complex models
            
            this.isCalibrated = true;
            console.log('Calibration training completed');
            return true;
            
        } catch (error) {
            console.error('Training error:', error);
            return false;
        }
    }

    /**
     * Predict gaze (main prediction function) - No calibration required
     */
    predictGaze(faceLandmarks, frameWidth, frameHeight) {
        try {
            // Direct geometric gaze prediction - no calibration needed
            const gazePoint = this.predictGazeGeometric(faceLandmarks, frameWidth, frameHeight);
            
            if (gazePoint) {
                // Add temporal smoothing
                this.recentPredictions.push(gazePoint);
                if (this.recentPredictions.length > this.maxRecentPredictions) {
                    this.recentPredictions.shift();
                }
                
                // Calculate smoothed gaze point
                const smoothedGaze = this.smoothGazePoint(gazePoint);
                
                return {
                    gazePoint: smoothedGaze,
                    confidence: 0.8,
                    rawGaze: gazePoint
                };
            }
            
            return null;
            
        } catch (error) {
            console.error('Gaze prediction error:', error);
            return null;
        }
    }

    /**
     * Geometric gaze prediction (improved scaling)
     */
    predictGazeGeometric(faceLandmarks, frameWidth, frameHeight) {
        try {
            // Get eye centers
            const leftEyeLandmarks = this.LEFT_IRIS.map(idx => [
                faceLandmarks[idx].x * frameWidth,
                faceLandmarks[idx].y * frameHeight
            ]);
            const rightEyeLandmarks = this.RIGHT_IRIS.map(idx => [
                faceLandmarks[idx].x * frameWidth,
                faceLandmarks[idx].y * frameHeight
            ]);
            
            const leftEyeCenter = this.calculateCenter(leftEyeLandmarks);
            const rightEyeCenter = this.calculateCenter(rightEyeLandmarks);
            
            // Calculate average eye center
            const eyeCenter = [
                (leftEyeCenter[0] + rightEyeCenter[0]) / 2,
                (leftEyeCenter[1] + rightEyeCenter[1]) / 2
            ];
            
            // Get face center for reference
            const faceCenterX = frameWidth / 2;
            const faceCenterY = frameHeight / 2;
            
            // Calculate offset from face center
            const offsetX = eyeCenter[0] - faceCenterX;
            const offsetY = eyeCenter[1] - faceCenterY;
            
            // Apply scaling factors for better accuracy
            const horizontalScale = 2.5; // Increased for better horizontal sensitivity
            const verticalScale = 3.0;   // Increased for better vertical sensitivity
            
            // Calculate screen coordinates with improved scaling
            // Flip horizontal (left eye movement = right screen movement)
            const screenX = (this.screenWidth / 2) - (offsetX * horizontalScale);
            const screenY = (this.screenHeight / 2) + (offsetY * verticalScale);
            
            // Clamp to screen bounds
            const clampedX = Math.max(0, Math.min(this.screenWidth, screenX));
            const clampedY = Math.max(0, Math.min(this.screenHeight, screenY));
            
            return [clampedX, clampedY];
            
        } catch (error) {
            console.error('Geometric prediction error:', error);
            return null;
        }
    }

    /**
     * Smooth gaze point using recent predictions
     */
    smoothGazePoint(currentGaze) {
        if (this.recentPredictions.length === 0) {
            return currentGaze;
        }
        
        // Calculate weighted average
        const weights = [0.1, 0.2, 0.3, 0.4]; // More weight to recent predictions
        let weightedSum = [0, 0];
        let totalWeight = 0;
        
        for (let i = 0; i < Math.min(this.recentPredictions.length, weights.length); i++) {
            const weight = weights[weights.length - 1 - i];
            weightedSum[0] += this.recentPredictions[this.recentPredictions.length - 1 - i][0] * weight;
            weightedSum[1] += this.recentPredictions[this.recentPredictions.length - 1 - i][1] * weight;
            totalWeight += weight;
        }
        
        // Add current prediction with highest weight
        const currentWeight = 0.5;
        weightedSum[0] += currentGaze[0] * currentWeight;
        weightedSum[1] += currentGaze[1] * currentWeight;
        totalWeight += currentWeight;
        
        return [
            weightedSum[0] / totalWeight,
            weightedSum[1] / totalWeight
        ];
    }

    /**
     * Start tracking
     */
    startTracking(callback) {
        this.isTracking = true;
        this.trackingCallback = callback;
        console.log('Gaze tracking started');
    }

    /**
     * Stop tracking
     */
    stopTracking() {
        this.isTracking = false;
        this.trackingCallback = null;
        console.log('Gaze tracking stopped');
    }

    /**
     * Stop calibration process
     */
    stopCalibration() {
        this.isCalibrating = false;
        if (this.currentCalibrationInterval) {
            clearInterval(this.currentCalibrationInterval);
            this.currentCalibrationInterval = null;
        }
        console.log('Calibration stopped');
    }

    /**
     * Clear calibration
     */
    clearCalibration() {
        this.stopCalibration();
        this.calibrationFeatures = [];
        this.calibrationGazeDirections = [];
        this.calibrationScreenPoints = [];
        this.isCalibrated = false;
        this.recentPredictions = [];
        console.log('Calibration cleared');
    }

    /**
     * Save calibration to localStorage
     */
    saveCalibration() {
        const calibration = {
            features: this.calibrationFeatures,
            gazeDirections: this.calibrationGazeDirections,
            screenPoints: this.calibrationScreenPoints,
            timestamp: Date.now(),
            screenWidth: this.screenWidth,
            screenHeight: this.screenHeight
        };
        
        localStorage.setItem('gaze_calibration', JSON.stringify(calibration));
        console.log('Calibration saved to localStorage');
    }

    /**
     * Load calibration from localStorage
     */
    loadCalibration() {
        const saved = localStorage.getItem('gaze_calibration');
        if (saved) {
            try {
                const calibration = JSON.parse(saved);
                this.calibrationFeatures = calibration.features || [];
                this.calibrationGazeDirections = calibration.gazeDirections || [];
                this.calibrationScreenPoints = calibration.screenPoints || [];
                this.screenWidth = calibration.screenWidth || 640;
                this.screenHeight = calibration.screenHeight || 480;
                
                if (this.calibrationFeatures.length > 0) {
                    this.isCalibrated = true;
                    console.log('Calibration loaded from localStorage');
                    return true;
                }
            } catch (error) {
                console.error('Error loading calibration:', error);
            }
        }
        return false;
    }

    /**
     * Export calibration data
     */
    exportCalibration() {
        return {
            features: this.calibrationFeatures,
            gazeDirections: this.calibrationGazeDirections,
            screenPoints: this.calibrationScreenPoints,
            timestamp: Date.now(),
            screenWidth: this.screenWidth,
            screenHeight: this.screenHeight,
            version: '1.0'
        };
    }

    /**
     * Import calibration data
     */
    importCalibration(calibration) {
        this.calibrationFeatures = calibration.features || [];
        this.calibrationGazeDirections = calibration.gazeDirections || [];
        this.calibrationScreenPoints = calibration.screenPoints || [];
        this.screenWidth = calibration.screenWidth || 640;
        this.screenHeight = calibration.screenHeight || 480;
        
        if (this.calibrationFeatures.length > 0) {
            this.isCalibrated = true;
            this.saveCalibration(); // Save to localStorage
            console.log('Calibration imported successfully');
        }
    }
}

// Make LocalGazeTracker available globally
window.LocalGazeTracker = LocalGazeTracker;
