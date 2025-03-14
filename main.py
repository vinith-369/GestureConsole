import sys
import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, 
                            QLineEdit, QComboBox, QSpinBox, QProgressBar, 
                            QGroupBox, QRadioButton, QMessageBox, QFileDialog,
                            QListWidget,QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from key_controller import KeyController


class Custom_Console(QMainWindow):
    def __init__(self):
        super().__init__()  
        self.DATA_DIR = "data"
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
        self.stand_dir = os.path.join(self.DATA_DIR, "stand")
        if not os.path.exists(self.stand_dir):
            os.makedirs(self.stand_dir)
            
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_value = 3
        
        self.is_collecting = False
        self.is_countdown_active = False
        self.data_collection_counter = 0
        self.current_class = ""
        self.dataset_size = 400  
        self.current_game = ""
        
        self.key_controller = KeyController()
        
        # Initialize variables for play mode
        self.is_playing = False
        self.play_pose = None
        
        self.initUI()
        self.prediction_history = []
        self.prediction_max_history = 3  # Reduced from 5 to be more responsive
        self.last_prediction = "stand"
        self.stand_bias = 2  # Add bias toward stand position
        self.prediction_confidence = {}
        self.last_action_time = time.time()
        self.action_cooldown = 1.0  # 1 second cooldown for actions
        self.confidence_threshold = {
            "stand": 30,  # Very low threshold to return to stand
            "action": 70   # Higher threshold for other actions
        }
    def smooth_prediction(self, raw_prediction):
        """Apply temporal smoothing with improved efficiency"""
        current_time = time.time()
        
        # Keep history at fixed length (use a more efficient data structure)
        if not hasattr(self, 'prediction_history'):
            self.prediction_history = []
        
        # Add new prediction to history
        self.prediction_history.append(raw_prediction)
        
        # Maintain fixed-length history
        if len(self.prediction_history) > self.prediction_max_history:
            self.prediction_history.pop(0)
        
        # Fast path for consistent predictions
        if all(pred == raw_prediction for pred in self.prediction_history):
            return raw_prediction, 100.0
        
        # Count occurrences using Counter for better performance
        from collections import Counter
        prediction_counts = Counter(self.prediction_history)
        
        # Add bias toward stand
        if "stand" in prediction_counts:
            prediction_counts["stand"] += self.stand_bias
        else:
            prediction_counts["stand"] = self.stand_bias
        
        # Get most common prediction
        smoothed_prediction, count = prediction_counts.most_common(1)[0]
        
        # Calculate confidence (subtract bias if it's "stand")
        original_count = count - (self.stand_bias if smoothed_prediction == "stand" else 0)
        confidence = (original_count / len(self.prediction_history)) * 100
        
        # Apply cooldown for actions
        time_since_last_action = current_time - self.last_action_time
        if time_since_last_action < self.action_cooldown and smoothed_prediction != "stand" and self.last_prediction != smoothed_prediction:
            # During cooldown, keep previous prediction
            return self.last_prediction, 100.0
        
        # Apply appropriate threshold based on gesture type
        threshold = self.confidence_threshold["stand"] if smoothed_prediction == "stand" else self.confidence_threshold["action"]
        
        # Update action time if this is a new action
        if smoothed_prediction != "stand" and self.last_prediction != smoothed_prediction and confidence >= threshold:
            self.last_action_time = current_time
            return smoothed_prediction, confidence
        elif confidence >= threshold:
            return smoothed_prediction, confidence
        else:
            # Keep previous prediction if confidence is too low
            return self.last_prediction, confidence
    def initUI(self):
        self.setWindowTitle("Custom Console")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(800, 600)  # Set minimum window size
        
        # Stack widget for switching between windows
        self.stack = QTabWidget()
        self.stack.setTabPosition(QTabWidget.West)
        
        # Create windows
        self.training_window = QWidget()
        self.playing_window = QWidget()
        
        self.setup_training_window()
        self.setup_playing_window()
        
        self.stack.addTab(self.training_window, "Train Gestures")
        self.stack.addTab(self.playing_window, "Play Game")
        
        # Set central widget
        self.setCentralWidget(self.stack)

    
    def setup_training_window(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Game name input
        game_layout = QHBoxLayout()
        game_layout.addWidget(QLabel("Game Name:"))
        self.game_input = QLineEdit()
        game_layout.addWidget(self.game_input)
        self.game_set_btn = QPushButton("Set Game")
        self.game_set_btn.clicked.connect(self.set_game)
        game_layout.addWidget(self.game_set_btn)
        layout.addLayout(game_layout)
        
        # Gesture input section
        gesture_group = QGroupBox("Gesture Configuration")
        gesture_layout = QVBoxLayout()
        
        # Action type selection
        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("Action Type:"))
        self.action_type_group = QGroupBox()
        action_radio_layout = QHBoxLayout()
        
        self.radio_press = QRadioButton("Pressing Key")
        self.radio_press.setChecked(True)
        self.radio_hold = QRadioButton("Holding Key")
        
        action_radio_layout.addWidget(self.radio_press)
        action_radio_layout.addWidget(self.radio_hold)
        self.action_type_group.setLayout(action_radio_layout)
        action_layout.addWidget(self.action_type_group)
        
        gesture_layout.addLayout(action_layout)
        
        # Key input
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Key:"))
        self.key_input = QLineEdit()
        key_layout.addWidget(self.key_input)
        gesture_layout.addLayout(key_layout)
        
        # Add gesture button
        button_layout = QHBoxLayout()
        self.add_gesture_btn = QPushButton("Add Gesture")
        self.add_gesture_btn.clicked.connect(self.add_gesture_class)
        self.add_gesture_btn.setEnabled(False)
        button_layout.addWidget(self.add_gesture_btn)
        
        # Train stand position button
        self.add_stand_btn = QPushButton("Add Stand Position")
        self.add_stand_btn.clicked.connect(self.add_stand_position)
        self.add_stand_btn.setEnabled(False)
        button_layout.addWidget(self.add_stand_btn)
        
        gesture_layout.addLayout(button_layout)
        
        gesture_group.setLayout(gesture_layout)
        layout.addWidget(gesture_group)
        
        # Data collection group
        collection_group = QGroupBox("Data Collection")
        collection_layout = QVBoxLayout()
        
        # Current classes list
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Gestures to Train:"))
        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self.select_class)
        class_layout.addWidget(self.class_list)
        collection_layout.addLayout(class_layout)
        
        # Camera feed - keep same size but ensure content is properly fit
        self.camera_label = QLabel("Camera Off")
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid #cccccc; font-size: 24px; background-color: black;")
        # Add size policy after creating the camera_label
        from PyQt5.QtWidgets import QSizePolicy
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        collection_layout.addWidget(self.camera_label)
        
        # Status and controls
        status_layout = QHBoxLayout()
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        collection_layout.addLayout(status_layout)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        buttons_layout.addWidget(self.start_camera_btn)
        
        self.collect_btn = QPushButton("Start Collection")
        self.collect_btn.clicked.connect(self.start_collection)
        self.collect_btn.setEnabled(False)
        buttons_layout.addWidget(self.collect_btn)
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        buttons_layout.addWidget(self.train_btn)
        
        collection_layout.addLayout(buttons_layout)
        
        collection_group.setLayout(collection_layout)
        layout.addWidget(collection_group)
        
        self.training_window.setLayout(layout)
    def setup_playing_window(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Game selection
        game_layout = QHBoxLayout()
        game_layout.addWidget(QLabel("Select Game:"))
        self.game_selector = QComboBox()
        self.update_game_selector()
        game_layout.addWidget(self.game_selector)
        layout.addLayout(game_layout)
        
        # Camera feed - keep same size but ensure content is properly fit
        self.play_camera_label = QLabel("Camera Off")
        self.play_camera_label.setFixedSize(640, 480)
        self.play_camera_label.setAlignment(Qt.AlignCenter)
        self.play_camera_label.setStyleSheet("border: 2px solid #cccccc; font-size: 24px; background-color: black;")
        # Add size policy after creating the camera label
        from PyQt5.QtWidgets import QSizePolicy
        self.play_camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.play_camera_label)
        
        # Result display
        result_layout = QHBoxLayout()
        result_layout.addWidget(QLabel("Detected Gesture:"))
        self.result_label = QLabel("None")
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007bff;")
        result_layout.addWidget(self.result_label)
        layout.addLayout(result_layout)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.play_camera_btn = QPushButton("Start Camera")
        self.play_camera_btn.clicked.connect(self.toggle_play_camera)
        controls_layout.addWidget(self.play_camera_btn)
        
        self.play_btn = QPushButton("Start Playing")
        self.play_btn.clicked.connect(self.toggle_playing)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)
        
        layout.addLayout(controls_layout)
        
        self.playing_window.setLayout(layout)

    
    def set_game(self):
        """Set current game and create game directory"""
        game_name = self.game_input.text().strip()
        if not game_name:
            QMessageBox.warning(self, "Input Error", "Please enter a game name!")
            return
        
        self.current_game = game_name
        
        # Create game directory
        game_dir = os.path.join(self.DATA_DIR, game_name)
        if not os.path.exists(game_dir):
            os.makedirs(game_dir)
        
        # Enable gesture adding
        self.add_gesture_btn.setEnabled(True)
        self.add_stand_btn.setEnabled(True)
        self.status_label.setText(f"Game set: {game_name}")
        
        # Disable game input
        self.game_input.setEnabled(False)
        self.game_set_btn.setEnabled(False)
        
        # Update class list
        self.update_class_list()
    
    def update_class_list(self):
        """Update the class list with available classes for the current game"""
        self.class_list.clear()
        if self.current_game and os.path.exists(os.path.join(self.DATA_DIR, self.current_game)):
            classes = [d for d in os.listdir(os.path.join(self.DATA_DIR, self.current_game)) 
                     if os.path.isdir(os.path.join(self.DATA_DIR, self.current_game, d))]
            for class_name in classes:
                self.class_list.addItem(class_name)
    
    def select_class(self, item):
        """Select a class from the list"""
        self.current_class = item.text()
    
    def update_game_selector(self):
        """Update the game selector with available games"""
        self.game_selector.clear()
        if os.path.exists(self.DATA_DIR):
            games = []
            for item in os.listdir(self.DATA_DIR):
                item_path = os.path.join(self.DATA_DIR, item)
                if os.path.isdir(item_path) and item != "stand" and any(os.path.isfile(os.path.join(item_path, f)) and f.endswith('.p') for f in os.listdir(item_path)):
                    games.append(item)
            
            if games:
                self.game_selector.addItems(games)
    
    def add_gesture_class(self):
        """Add a new gesture class based on user input"""
        action_type = "press" if self.radio_press.isChecked() else "hold"
        key = self.key_input.text().strip()
        
        if not key:
            QMessageBox.warning(self, "Input Error", "Please enter a key!")
            return
        
        class_name = f"{action_type}_{key}"
        
        # Create directory for the class
        game_class_dir = os.path.join(self.DATA_DIR, self.current_game, class_name)
        if not os.path.exists(game_class_dir):
            os.makedirs(game_class_dir)
            QMessageBox.information(self, "Success", f"Added gesture class: {class_name}")
            
            # Update class list
            self.update_class_list()
            self.key_input.clear()
        else:
            QMessageBox.warning(self, "Error", f"Class {class_name} already exists!")
    
    def add_stand_position(self):
        """Add stand position for the current game"""
        game_stand_dir = os.path.join(self.DATA_DIR, self.current_game, "stand")
        if not os.path.exists(game_stand_dir):
            os.makedirs(game_stand_dir)
            QMessageBox.information(self, "Success", "Added stand position class")
            
            # Update class list
            self.update_class_list()
        else:
            QMessageBox.warning(self, "Error", "Stand position class already exists!")
    
    def toggle_camera(self):
        """Start or stop the camera in training window"""
        if self.timer.isActive():
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.start_camera_btn.setText("Start Camera")
            self.collect_btn.setEnabled(False)
            self.camera_label.setText("Camera Off")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open camera!")
                return
                
            # Set camera to capture at higher resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
            self.timer.start(30)  # 30ms refresh (approx 30 FPS)
            self.start_camera_btn.setText("Stop Camera")
            self.collect_btn.setEnabled(True)
    
    def toggle_play_camera(self):
        """Start or stop the camera in play window"""
        if self.timer.isActive():
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.play_camera_btn.setText("Start Camera")
            self.play_btn.setEnabled(False)
            self.play_camera_label.setText("Camera Off")
        else:
            # Check if game is selected
            selected_game = self.game_selector.currentText()
            if not selected_game:
                QMessageBox.warning(self, "Warning", "Please select a game first!")
                return
                
            # Check if model exists
            model_path = os.path.join(self.DATA_DIR, selected_game, f"{selected_game}_model.p")
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Warning", f"No model found for {selected_game}!")
                return
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open camera!")
                return
                
            self.timer.start(30)  # 30ms refresh (approx 30 FPS)
            self.play_camera_btn.setText("Stop Camera")
            self.play_btn.setEnabled(True)
    
    def update_frame(self):
        """Update the camera frame with pose detection and data collection"""
        if self.cap is None:
            return
                
        ret, frame = self.cap.read()
        if not ret:
            return
                
        # Flip the frame horizontally for a selfie-view
        frame = cv2.flip(frame, 1)
        
        # Get the actual frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate aspect ratio to maintain proportions
        if self.stack.currentIndex() == 0:  # Training window
            camera_label_width = self.camera_label.width()
            camera_label_height = self.camera_label.height()
        else:  # Play window
            camera_label_width = self.play_camera_label.width()
            camera_label_height = self.play_camera_label.height()
        
        aspect_ratio = frame_width / frame_height
        display_width = camera_label_width
        display_height = int(display_width / aspect_ratio)
        
        if display_height > camera_label_height:
            display_height = camera_label_height
            display_width = int(display_height * aspect_ratio)
        
        # For data collection
        if self.stack.currentIndex() == 0 and self.is_collecting and not self.is_countdown_active:
            # Process frame for data collection
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Save image for training
                img_path = os.path.join(self.DATA_DIR, self.current_game, self.current_class, 
                                    f"sample_{self.data_collection_counter}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Update counter and progress
                self.data_collection_counter += 1
                progress = int((self.data_collection_counter / self.dataset_size) * 100)
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"Collecting {self.current_class}: {self.data_collection_counter}/{self.dataset_size}")
                
                # Check if we've collected enough data for current class
                if self.data_collection_counter >= self.dataset_size:
                    self.is_collecting = False
                    self.collect_btn.setText("Start Collection")
                    self.status_label.setText(f"Collection complete for {self.current_class}")
                    
                    # Auto-transition to next class
                    self.auto_transition_to_next_class()
                    return
        
        # For countdown display
        if self.is_countdown_active:
            # Draw countdown on frame
            cv2.putText(frame, str(self.countdown_value), 
                    (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
        
        # For play window
        if self.stack.currentIndex() == 1 and self.is_playing:
            # Process frame for landmark detection
            if not hasattr(self, 'frame_counter'):
                self.frame_counter = 0
            
            self.frame_counter += 1
            if self.frame_counter % 2 != 0:  # Process every other frame
                # Display frame only without processing
                display_frame = cv2.resize(frame, (display_width, display_height))
                h, w, c = display_frame.shape
                qimg = QImage(display_frame.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                self.play_camera_label.setPixmap(QPixmap.fromImage(qimg))
                self.play_camera_label.setAlignment(Qt.AlignCenter)
                return
                
            # Resize frame for faster processing while keeping a copy of original for display
            small_frame = cv2.resize(frame, (320, 240))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Use cached pose detector
            if not hasattr(self, 'play_pose') or self.play_pose is None:
                self.play_pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # Use simplest model for speed
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                    
            results = self.play_pose.process(frame_rgb)
            
            # Process landmarks if detected
            prediction = "stand"  # Default to stand
            confidence = 100.0
            
            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Extract features and predict
                try:
                    # Extract keypoints using vectorized operations
                    landmarks = results.pose_landmarks.landmark
                    data_aux = []
                    x_ = [landmark.x for landmark in landmarks]
                    y_ = [landmark.y for landmark in landmarks]
                    
                    min_x, min_y = min(x_), min(y_)
                    
                    # Normalize coordinates in a faster way
                    for landmark in landmarks:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)
                    
                    # Make prediction
                    prediction = self.model.predict([data_aux])[0]
                    
                    # Apply improved smoothing
                    smoothed_prediction, confidence = self.smooth_prediction(prediction)
                    prediction = smoothed_prediction
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    # Default to stand position on error
            
            # Handle key actions outside the landmark detection block
            self.handle_prediction(prediction, confidence)
                    
            # Show prediction on frame
            cv2.putText(frame, f"Gesture: {prediction}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize the frame for display while maintaining aspect ratio
        display_frame = cv2.resize(frame, (display_width, display_height))
        h, w, c = display_frame.shape
        qimg = QImage(display_frame.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
        
        if self.stack.currentIndex() == 0:  # Training window
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))
            self.camera_label.setAlignment(Qt.AlignCenter)
        elif self.stack.currentIndex() == 1:  # Play window
            self.play_camera_label.setPixmap(QPixmap.fromImage(qimg))
            self.play_camera_label.setAlignment(Qt.AlignCenter)
    def handle_prediction(self, prediction, confidence):
        """Handle key presses based on gesture prediction - separated for performance"""
        # Update result label
        self.result_label.setText(f"{prediction} ({confidence:.1f}%)")
        
        # Handle key state transitions
        if prediction != self.last_prediction:
            # We're changing gestures, release previous keys if needed
            if self.last_prediction != "stand":
                try:
                    old_action, old_key = self.last_prediction.split("_")
                    self.key_controller.release_key(old_key)
                except ValueError:
                    # Handle case where prediction isn't in expected format
                    self.key_controller.release_all_keys()
        
        # Execute actions based on prediction
        if prediction != "stand":
            try:
                action, key = prediction.split("_")
                if action == "press":
                    # For press actions, press and release quickly
                    self.key_controller.press_key(key)
                    # Use a shorter delay for better responsiveness
                    QTimer.singleShot(50, lambda k=key: self.key_controller.release_key(k))
                elif action == "hold":
                    # For hold actions, just make sure key is pressed
                    self.key_controller.hold_key(key)
            except ValueError:
                # Handle case where prediction isn't in expected format
                print(f"Invalid prediction format: {prediction}")
        else:
            # Release all keys when stand position detected
            self.key_controller.release_all_keys()
        
        # Update last prediction
        self.last_prediction = prediction
    def update_countdown(self):
        """Update the countdown timer for data collection"""
        self.countdown_value -= 1
        
        if self.countdown_value <= 0:
            self.countdown_timer.stop()
            self.is_countdown_active = False
            # Start actual collection after countdown
            self.status_label.setText(f"Collecting {self.current_class}...")
            # Reset countdown for next time
            self.countdown_value = 3
    def auto_transition_to_next_class(self):
        """Automatically transition to the next gesture class for data collection"""
        # Get all available classes
        if not self.current_game:
            return
            
        game_dir = os.path.join(self.DATA_DIR, self.current_game)
        if not os.path.exists(game_dir):
            return
            
        classes = [d for d in os.listdir(game_dir) if os.path.isdir(os.path.join(game_dir, d))]
        
        # Find current class index
        if self.current_class not in classes:
            self.train_btn.setEnabled(True)  # Enable training if all classes are done
            return
            
        current_idx = classes.index(self.current_class)
        
        # Find next class that hasn't been fully collected yet
        next_class = None
        for i in range(current_idx + 1, len(classes)):
            class_dir = os.path.join(game_dir, classes[i])
            # Check if this class needs more samples
            sample_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if sample_count < self.dataset_size:
                next_class = classes[i]
                break
        
        if next_class:
            # Select the next class in the list and start collection
            self.current_class = next_class
            self.class_list.setCurrentItem(self.class_list.findItems(next_class, Qt.MatchExactly)[0])
            
            # Reset collection counter
            self.data_collection_counter = 0
            
            # Start collection for the next class after a brief delay
            QMessageBox.information(self, "Next Class", f"Ready to collect data for: {next_class}")
            QTimer.singleShot(1000, self.start_collection)
        else:
            # All classes collected, enable training
            self.train_btn.setEnabled(True)
            QMessageBox.information(self, "Collection Complete", "All gesture data collected. Ready to train the model.")
    
    def start_collection(self):
        if self.is_collecting:
            # Stop collection
            self.is_collecting = False
            self.collect_btn.setText("Start Collection")
            self.status_label.setText("Collection stopped")
        else:
            # Check if class is selected
            if not self.current_class:
                item = self.class_list.currentItem()
                if item:
                    self.current_class = item.text()
                else:
                    QMessageBox.warning(self, "Error", "Please select a gesture class from the list!")
                    return
            
            # Check if directory exists
            class_dir = os.path.join(self.DATA_DIR, self.current_game, self.current_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            # Reset counters
            self.data_collection_counter = 0
            self.progress_bar.setValue(0)
            
            # Set the status
            self.status_label.setText("Get ready...")
            
            # Start countdown
            self.is_collecting = True
            self.is_countdown_active = True
            self.collect_btn.setText("Stop Collection")
            self.countdown_value = 3
            self.countdown_timer.start(1000)  # 1 second intervals
    
    def start_training(self):
        """Extract keypoints and train the model"""
        if not self.current_game:
            QMessageBox.warning(self, "Error", "No game selected!")
            return
        
        # Show progress dialog
        self.progress_bar.setValue(0)
        self.status_label.setText("Extracting keypoints...")
        QApplication.processEvents()
        
        # Track image paths for later deletion
        all_image_paths = []
        
        # Extract keypoints and train model
        try:
            game_dir = os.path.join(self.DATA_DIR, self.current_game)
            data = []
            labels = []
            
            # List all classes
            classes = [d for d in os.listdir(game_dir) if os.path.isdir(os.path.join(game_dir, d))]
            
            if not classes:
                QMessageBox.warning(self, "Error", "No gesture classes found!")
                return
            
            # Count total files for progress tracking
            total_files = 0
            for dir_ in classes:
                dir_path = os.path.join(game_dir, dir_)
                total_files += len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            processed_files = int(0)
            
            # Process each class
            for dir_ in classes:
                img_files = [f for f in os.listdir(os.path.join(game_dir, dir_)) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                if not img_files:
                    continue
                    
                for img_path in img_files:
                    full_img_path = os.path.join(game_dir, dir_, img_path)
                    all_image_paths.append(full_img_path)  # Store path for later deletion
                    
                    data_aux = []
                    x_ = []
                    y_ = []
                    
                    # Read and process image
                    img = cv2.imread(full_img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(img_rgb)
                    
                    # Extract pose landmarks
                    if results.pose_landmarks:
                        for landmark in results.pose_landmarks.landmark:
                            x_.append(landmark.x)
                            y_.append(landmark.y)
                        
                        # Normalize coordinates
                        for landmark in results.pose_landmarks.landmark:
                            data_aux.append(landmark.x - min(x_))
                            data_aux.append(landmark.y - min(y_))
                        
                        data.append(data_aux)
                        labels.append(dir_)
                    
                    # Update progress
                    processed_files += 1
                    self.progress_bar.setValue(int((processed_files / total_files) * 50))  # 50% for extraction
                    QApplication.processEvents()
            
            self.status_label.setText("Training model...")
            QApplication.processEvents()
            
            # Check if data is sufficient
            if len(data) < 10:
                QMessageBox.warning(self, "Warning", "Not enough data samples to train a reliable model!")
                return
            
            # Convert to numpy arrays
            data = np.asarray(data)
            labels = np.asarray(labels)
            
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                data, labels, test_size=0.2, shuffle=True, stratify=labels
            )
            
            # Train model
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            
            # Evaluate model
            y_predict = model.predict(x_test)
            score = accuracy_score(y_predict, y_test)
            
            # Save model
            model_path = os.path.join(game_dir, f"{self.current_game}_model.p")
            with open(model_path, 'wb') as f:
                pickle.dump({'model': model, 'labels': list(set(labels))}, f)
            
            # Delete all images now that the model is trained
            self.status_label.setText("Cleaning up training data...")
            self.progress_bar.setValue(75)
            QApplication.processEvents()
            
            # Delete images
            for img_path in all_image_paths:
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")
            
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Model trained with {score * 100:.2f}% accuracy")
            
            # Update game selector
            self.update_game_selector()
            
            # Reset game setup for new game
            self.game_input.setEnabled(True)
            self.game_set_btn.setEnabled(True)
            self.game_input.clear()
            self.current_game = ""
            self.train_btn.setEnabled(False)
            self.add_gesture_btn.setEnabled(False)
            self.add_stand_btn.setEnabled(False)
            
            QMessageBox.information(self, "Success", 
                                f"Model trained with {score * 100:.2f}% accuracy and training images deleted")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train model: {str(e)}")
    def toggle_playing(self):
        """Start or stop playing with gesture control"""
        if self.is_playing:
            try:
                # Set this flag first to prevent further actions while stopping
                self.is_playing = False
                self.play_btn.setText("Start Playing")
                self.result_label.setText("None")
                
                # Process events to update UI before potentially slow operations
                QApplication.processEvents()
                
                # Release all keys with timeout protection
                try:
                    # Use a QTimer with short timeout to release keys
                    # This prevents UI freezing if key release hangs
                    self.key_controller.held_keys = []  # Clear held keys list immediately
                    QTimer.singleShot(100, self.key_controller.release_all_keys)
                except Exception as e:
                    print(f"Error releasing keys: {e}")
                
                # Reset prediction variables
                self.prediction_history = []
                self.last_prediction = "stand"
                self.last_action_time = time.time() - 10  # Reset cooldown
            except Exception as e:
                print(f"Error stopping play mode: {e}")
        else:
            selected_game = self.game_selector.currentText()
            if not selected_game:
                QMessageBox.warning(self, "Warning", "Please select a game!")
                return
                
            # Load model
            try:
                model_path = os.path.join(self.DATA_DIR, selected_game, f"{selected_game}_model.p")
                model_data = pickle.load(open(model_path, 'rb'))
                self.model = model_data['model']
                self.is_playing = True
                self.play_btn.setText("Stop Playing")
                
                # Reset key states
                QTimer.singleShot(100, self.key_controller.release_all_keys)
                
                # Reset prediction variables
                self.prediction_history = []
                self.last_prediction = "stand"
                self.last_action_time = time.time() - 10  # Ensure no initial cooldown
                self.frame_counter = 0  # Initialize frame counter
                
                # Set more responsive parameters
                self.prediction_max_history = 2  # Smaller history for faster response
                self.action_cooldown = 0.5  # Shorter cooldown between actions
                self.confidence_threshold = {
                    "stand": 20,  # Lower threshold to return to stand
                    "action": 60   # Lower threshold for actions
                }
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        
        # Clear held keys list immediately
        if hasattr(self, 'key_controller') and hasattr(self.key_controller, 'held_keys'):
            self.key_controller.held_keys = []
        
        # Give time for UI to update before potentially slow operations
        QApplication.processEvents()
        
        # Use short timeout for releasing keys
        try:
            QTimer.singleShot(100, lambda: None)
        except Exception:
            pass
    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        if hasattr(self, 'camera_label') and hasattr(self, 'play_camera_label'):
            window_width = self.width()
            window_height = self.height()
            new_width = min(int(window_width * 0.6), int(window_height * 0.6 * 4/3))
            new_height = int(new_width * 3/4)
            
            # Update camera labels
            self.camera_label.setFixedSize(new_width, new_height)
            self.play_camera_label.setFixedSize(new_width, new_height)
            
            # Update the layout to reflect the changes
            if hasattr(self, 'training_window') and self.training_window.layout():
                self.training_window.layout().update()
            if hasattr(self, 'playing_window') and self.playing_window.layout():
                self.playing_window.layout().update()
if __name__ == "__main__":  # Fixed syntax error here
    app = QApplication(sys.argv)
    app.setStyle('Macintosh')  # Modern look across platforms
    window = Custom_Console()
    window.show()
    sys.exit(app.exec_())