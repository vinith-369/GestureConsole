# Gesture Console

**Gesture Console** is a computer vision-based application that allows users to control games using hand and body poses. It leverages **MediaPipe** for pose detection and stores trained models using **Pickle**, so users can play games without retraining.

## Features
- Control games using gestures
- Train once and play anytime (model stored using Pickle)
- Supports **Windows & Mac**
- Easy-to-use GUI

## Installation
### Windows
1. **Download the EXE file** from the [releases](https://github.com/vinith-369/GestureConsole/releases) section.
2. **Run the EXE file** (no installation required).

### Mac (Manual Setup)
1. Clone the repository:
   ```sh
   git clone https://github.com/vintih/GestureConsole.git
   cd GestureConsole
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python app.py
   ```

## Usage
1. **Train your gestures**
   - Open Gesture Console
   - Follow on-screen instructions to train poses
   - The trained model is saved automatically
2. **Control a game**
   - Run the application
   - Load your saved model
   - Start using gestures to control the game

## Dependencies
- Python 3.9+
- MediaPipe
- OpenCV
- NumPy
- Pickle

## Contributing
Feel free to fork this repository and contribute improvements!

## License
This project is licensed under the MIT License.

---
Made with ❤️ by **Vinith**

