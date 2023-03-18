# Artificial-Sign-Language-ASL-

Contributors to the Project: 
- Pooja Patel
- Mohsin Shah
- Ananya Shekhawat
- Emma Azzi
- Cyril Bou-Harb
- Hassan Shah

Our program connects to our laptop's camera so we can track our hands movements, thus, we were able to translate American Sign Language into English text
in a live video feed using artificial intelligence. We used specific libraries in python to open our laptop's camera and track our hands. OpenCV and
MediaPipe allowed us to collect sensory data that was captured with our computer's camera. With both of these libraries, we could track our hands when they
appeared on camera by getting access to coordinate points on our hands. The third library we used is called TensorFlow. TensorFlow is a machine learning 
library that scales the data we are collecting and accelerates training our AI. We implemented a way to create a image crop around the our hands by 
calculating the distance between each coordinate points on our hands. With an accurate image crop, we could save images of our hands in certain positions
by pressing a key on the keyboard since we created a save key in our program. With hundreds and hundreds of saved images in a folder corresponding to the
sign we were showing in the camera, we uploaded the images to Google's Teachable Machine, so we could easily train our AI to translate these images of the
ASL sign to the correct word or phrase in English. Thus, when we ran our program, our AI was able to translate ASL to English text in real time video feed.
