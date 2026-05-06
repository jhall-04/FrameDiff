from flask import Flask, Response
import time
import cv2

app = Flask(__name__)

def generate(loop=True):
    while loop:
        cap = cv2.VideoCapture("demo/output.mp4")  # replace with your video path
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
        time.sleep(1/60)
@app.route('/')
def index():
    return '''
        <html>
        <body style="background:black; display:flex; justify-content:center; align-items:center; height:100vh;">
            <img src="/video" style="max-width:100%">
        </body>
        </html>
    '''

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)
# Open http://localhost:5000/video in your Windows browser