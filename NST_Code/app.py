import os
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Copy demo files on startup
demo_files = ['brad_pitt.jpg', 'sketch.png', 'picasso_seated_nude_hr.jpg', 'la_muse.jpg']
examples_dir = os.path.join(BASE_DIR, 'examples')
for fname in demo_files:
    src = os.path.join(examples_dir, fname)
    dst = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    if os.path.exists(src) and not os.path.exists(dst):
        try:
            shutil.copy(src, dst)
        except Exception:
            pass

# ── ONNX Runtime sessions (loaded once at startup) ────────────────────────────
_encoder_sess = ort.InferenceSession(
    os.path.join(BASE_DIR, 'encoder.onnx'),
    providers=['CPUExecutionProvider']
)
_decoder_sess = ort.InferenceSession(
    os.path.join(BASE_DIR, 'decoder.onnx'),
    providers=['CPUExecutionProvider']
)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def _preprocess(pil_img, size=512):
    """Resize shortest side to `size`, convert to float32 NCHW in [0,1]."""
    w, h = pil_img.size
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0   # HWC [0,1]
    arr = arr.transpose(2, 0, 1)[np.newaxis]             # NCHW
    return arr


def _adain(content_feat, style_feat, eps=1e-5):
    """Adaptive Instance Normalization in pure NumPy."""
    c_mean = content_feat.mean(axis=(2, 3), keepdims=True)
    c_std  = content_feat.std(axis=(2, 3), keepdims=True) + eps
    s_mean = style_feat.mean(axis=(2, 3), keepdims=True)
    s_std  = style_feat.std(axis=(2, 3), keepdims=True) + eps
    return s_std * (content_feat - c_mean) / c_std + s_mean


def style_transfer(content_pil, style_pil, alpha):
    """Full pipeline using ONNX Runtime."""
    content_arr = _preprocess(content_pil)
    style_arr   = _preprocess(style_pil)

    # Encode
    content_feats = _encoder_sess.run(['features'], {'image': content_arr})[0]
    style_feats   = _encoder_sess.run(['features'], {'image': style_arr})[0]

    # AdaIN + alpha blend
    stylized_feats = _adain(content_feats, style_feats)
    blended        = alpha * stylized_feats + (1.0 - alpha) * content_feats

    # Decode
    output = _decoder_sess.run(['image'], {'features': blended.astype(np.float32)})[0]

    # Post-process: NCHW → HWC uint8
    output = output.squeeze(0).transpose(1, 2, 0)
    output = np.clip(output, 0.0, 1.0)
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path   = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image   = Image.open(style_path).convert('RGB')
                alpha         = float(form.alpha.data)
                stylized      = style_transfer(content_image, style_image, alpha)
                result_filename = 'stylized_' + content_filename
                stylized.save(os.path.join(app.config['UPLOAD_FOLDER'], result_filename))
                result_image = result_filename
            except Exception as e:
                error = str(e)
    else:
        if request.method == 'POST':
            if not content_filename:
                error = 'Please select or upload content image'
            elif not style_filename:
                error = 'Please select or upload style image'

    return render_template('index.html', form=form, result_image=result_image,
                           content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)