import io
from pathlib import Path

import torch
import constants
from PIL import Image, ImageOps
from flask import Flask, request, redirect, render_template, url_for, flash

from models.yolo import attempt_load
from utils.general import set_logging
from utils.torch_utils import select_device

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/", methods=["POST"])
def predict():
    if request.method == "POST":
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

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        transposed_image = ImageOps.exif_transpose(img)

        # load model
        model = get_custom_model()

        # set model params
        model.conf = 0.4
        model.iou = 0.3

        # compute and save result
        results = model(transposed_image, size=480)
        results.save(constants.SAVE_DIR)

        return render_template("index.html", filename=constants.DEFAULT_FILE_NAME)


@app.route('/display/<filename>', methods=["GET"])
def display_image(filename):
    return redirect(url_for('static', filename="saved/" + filename))


@app.route("/try/<file_name>", methods=["GET"])
def try_it(file_name):
    # load model
    model = get_custom_model()

    img = Image.open(constants.IMAGE_BASE_PATH + file_name)

    results = model(img, size=640)
    results.save(constants.SAVE_DIR)

    return render_template("index.html", filename=file_name)


def get_custom_model(autoshape=True, verbose=True, device=None):
    # enable logging
    set_logging(verbose=verbose)

    model_path = Path(constants.MODEL_PATH)
    model = attempt_load(model_path, map_location=torch.device('cpu'))
    if autoshape:
        model = model.autoshape()
    device = select_device('0' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    return model.to(device)


if __name__ == "__main__":
    print("name")
    app.run(debug=True)
