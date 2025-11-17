✅ 1. Create a virtual environment
python -m venv maitri_env


Activate it:

Windows:
maitri_env\Scripts\activate

✅ 2. Install dependencies

Since you uploaded only the model and webcam file, collaborators just need a few libraries:

pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib


(If you want, I can generate a requirements.txt file.)

✅ 3. Run your webcam emotion detection

If your webcam script is named webcam_emotion.py:

python webcam_emotion.py
