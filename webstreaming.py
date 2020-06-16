# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import time
import cv2
from detect import detect
import os

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
		
def generate():
	cap = cv2.VideoCapture('USA vs Brazil - Women-s Beach Volleyball - Highlights Nanjing 2014 Youth Olympic Games.mp4')
	fps = int(cap.get(cv2.CAP_PROP_FPS))

	weightsPath = os.path.join('yolov3_my2_6000.weights')
	configPath = os.path.join('yolov3_my2.cfg')
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	
	# loop over frames from the output stream
	start_time = int(time.time() * 1000)
	frame_number = 0
	while True:
		frame_number += 1
		current_time = int(time.time() * 1000)

		wait_time = 1000./fps * frame_number + start_time - current_time
		if wait_time > 0:
			time.sleep(wait_time/1000)
		ret, frame = cap.read()
		if ret:
			frame=cv2.resize(frame,None,fx=0.5,fy=0.5,
        		interpolation=cv2.INTER_AREA)

			frame = detect(net, frame)

			(flag, encodedImage) = cv2.imencode(".jpg", frame)

			# ensure the frame was successfully encoded
			if not flag:
				continue
			
			yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
				encodedImage.tobytes() + b'\r\n')
		else:
			break
	cap.release()
		

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)