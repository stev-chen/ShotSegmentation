# csc420_final

# Requirements
Make sure to run on Python 3

Need to install moviepy to cut video:

pip3 install ez_setup
pip3 install moviepy

Need numpy and cv2, if not installed as well.

To run be sure run.sh has execute permissions.

sudo ./run.sh videofilepath framerate threshold

Where videofilepath is the path to the video, framerate is the framerate to process at (does not have to be the same as video frame rate, lower framerates will have better runtime), and threshold is the ratio threshold to use (decrease if missing some shot detection).

I had best results with framerate = 6, threshold = 9

Here's an example run, with the included sample clip from the TV show Suits.

sudo ./run.sh suitscut.mp4 6 9
