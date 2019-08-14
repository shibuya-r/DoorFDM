from flask import Flask, Response, render_template, request
from camera import VideoCamera
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import platform
import interactive_detection

app = Flask(__name__)
logger = getLogger(__name__)

basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

is_async_mode = True
is_face_detection = True
is_head_pose_detection = False
is_facial_landmarks_detection = False


def gen(camera):
    while True:
        frame = camera.get_frame(is_async_mode, is_face_detection,is_head_pose_detection)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    if platform.system() == 'Linux':
        no_v4l = False
    else:
        no_v4l = True
    camera = VideoCamera(detections, no_v4l)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    # arg parse
    args = interactive_detection.build_argparser().parse_args()
    devices = [args.device,args.device_head_pose]
    models = [args.model_face,  args.model_head_pose, ]
    if "CPU" in devices and args.cpu_extension is None:
        print(
            "\nPlease try to specify cpu extensions library path in demo's command line parameters using -l "
            "or --cpu_extension command line argument")
        sys.exit(1)

    # Create detectors class instance
    detections = interactive_detection.Detections(
        devices, models, args.cpu_extension, args.plugin_dir,
        args.prob_threshold, args.prob_threshold_face, is_async_mode)
    models = detections.models  # Get models to display WebUI.

    app.run(host='0.0.0.0', threaded=True)
