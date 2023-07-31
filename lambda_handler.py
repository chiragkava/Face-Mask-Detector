import awsgi
from flask import (
    Flask,
    request,
    render_template, redirect, flash
)

from constants import constants
from service.face_mask_service import try_on_sample_image, process_image

app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def predict():
    # validate basic checks
    if "file" not in request.files:
        return redirect('/')
    file = request.files["file"]
    if not file:
        flash('Please select a file')
        return redirect('/')

    # validate if its image or not
    if file.content_type not in constants.ALLOWED_IMAGE_TYPE:
        flash("Image format not supported. Supported formats: {}".format(constants.ALLOWED_IMAGE_TYPE))
        return redirect('/')

    return process_image(file)


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')


@app.route("/try_it", methods=["GET"])
def try_it():
    file_name = request.args.get('file_name')
    return try_on_sample_image(file_name)


def lambda_handler(event, context):
    # For logging
    print(event)
    response = awsgi.response(app, event, context, base64_content_types=constants.ALLOWED_IMAGE_TYPE)
    return response
