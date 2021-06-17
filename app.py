import io
import os
from PIL import Image,ImageOps
import torch
from pathlib import Path
from models.yolo import Model, attempt_load
from utils.general import set_logging
from utils.torch_utils import select_device
from flask import Flask,request, redirect,render_template, url_for,flash


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect('/')
        file = request.files["file"]
        if not file:
            flash('Please select a file')
            return redirect('/')

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        new_img = ImageOps.exif_transpose(img)
        model = custom(path='last.pt')
        model.conf = 0.4  
        model.iou = 0.3 
        results = model(new_img, size=480)
        filename="image0.jpg"
        print("inside predict:",filename)
        results.save("static/saved")  
        return render_template("index.html",filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="saved/"+filename))

@app.route("/try/<name>")
def try_it(name):
    filename = name
    print("inside try: ",filename)
    model = custom(path="last.pt")
    img = Image.open("static/sample/"+name)
    results = model(img,size=640)
    print("inside try")
    results.save("static/saved")
    return render_template("index.html",filename=filename)

def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    set_logging(verbose=verbose)

    fname = Path(name).with_suffix('.pt') 
    model = attempt_load(fname, map_location=torch.device('cpu')) 
    if autoshape:
        model = model.autoshape() 
    device = select_device('0' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
    return model.to(device)
    
    
def custom(path='path/to/model.pt', autoshape=True, verbose=True, device=None):
    return _create(path, autoshape=autoshape, verbose=verbose, device=device)


if __name__ == "__main__":
    print("name")
    app.run(debug=True)