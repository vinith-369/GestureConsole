# Gesture Console

**Gesture Console** is a computer vision-based application that allows users to control games using hand and body poses. It leverages **MediaPipe** for pose detection and stores trained models using **Pickle**, so users can play games without retraining.

## Features
- Control games using gestures
- Train once and play anytime (model stored using Pickle)
- Supports **Windows & Mac**
- Easy-to-use GUI

## Installation
### Windows
1. **Download the EXE file** from the [releases](https://github.com/vintih/GestureConsole/releases).
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
1. **Set up your game**  
   - Open Gesture Console  
   - Enter the name of your game  

2. **Train your gestures**  
   - Add a **stand pose** (mandatory)  
   - Assign **hold and tap buttons** for gestures  
   - Select a key, turn on the camera, and train the model  

3. **Play the game**  
   - Go to **Play Game**  
   - Select your trained game  
   - Start the camera and play using gestures!  

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
