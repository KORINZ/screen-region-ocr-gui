# Screen Region OCR GUI

This Python application allows users to perform Optical Character Recognition (OCR) on a selected region of the screen or an imported image. The GUI is built using the CustomTkinter library and supports various image preprocessing options to enhance OCR accuracy.

## Features

- Select a region of the screen to capture and perform OCR
- Import an image and perform OCR
- Preprocess images to improve OCR accuracy
- Display input and output images
- Adjustable settings, including language, preprocessing parameters, and shortcuts

## Dependencies

- Python 3.6 or higher
- pytesseract
- Pillow (PIL)
- OpenCV (cv2)
- numpy
- pyautogui
- pyperclip
- customtkinter

## Installation

1. Clone the repository:

```
git clone https://github.com/korinz/screen_region_ocr_gui.git
```

2. Change to the project directory:

```
cd screen_region_ocr_gui
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Set the path to the Tesseract executable in `main.py`.

## Usage

1. Run the main script:

```
python main.py
```

2. Use the GUI to take a screenshot, import an image, and perform OCR on the selected region or imported image.

3. Adjust settings and preprocessing parameters as needed.

4. View the input and output images in the GUI.

5. The OCR result will be printed in the terminal.

## License

This project is licensed under the [MIT License](LICENSE).
