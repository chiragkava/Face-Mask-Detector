import base64
import io
from pathlib import Path

import torch
from PIL import Image, ImageOps
from flask import (render_template)

from constants import constants
from models.yolo import attempt_load
from utils.general import set_logging
from utils.torch_utils import select_device

IMAGE_SIZE = 640
HOME_TEMPLATE = "index.html"


def try_on_sample_image(file_name):
    # Get the trained model
    model = get_custom_model()
    img = Image.open(constants.IMAGE_BASE_PATH + file_name)

    results = model(img, size=IMAGE_SIZE)
    results.save(save_dir=constants.TEMP_DIR)

    b64_image = get_b64_image(constants.TEMP_DIR + file_name)
    return render_template(HOME_TEMPLATE, filename=file_name, b64_image=b64_image)


def get_custom_model(auto_shape=True, verbose=True, device=None):
    # enable logging
    set_logging(verbose=verbose)

    model_path = Path(constants.MODEL_PATH)
    model = attempt_load(model_path, map_location=torch.device('cpu'))
    if auto_shape:
        model = model.autoshape()
    device = select_device('0' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    return model.to(device)


def get_b64_image(path):
    # convert image to b64
    image = Image.open(path)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)

    return base64.encodebytes(image_bytes.getvalue()).decode()


def process_image(file):
    # convert multipart form data to image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    transposed_image = ImageOps.exif_transpose(img)

    # load model
    model = get_custom_model()

    # set model params
    set_model_parameters(model)

    # compute and save result
    results = model(transposed_image, size=IMAGE_SIZE)
    results.save(save_dir=constants.TEMP_DIR)

    b64_image = get_b64_image(constants.TEMP_DIR + constants.DEFAULT_FILE_NAME)
    return render_template(HOME_TEMPLATE, filename=constants.DEFAULT_FILE_NAME, b64_image=b64_image)


def set_model_parameters(model):
    model.conf = 0.4
    model.iou = 0.3

