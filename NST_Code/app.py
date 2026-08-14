import os
import uuid
import time
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, send_from_directory, make_response
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.config['SECRET_KEY'] = 'supersecretkey'

# Use /tmp on Vercel/serverless environments where /var/task is read-only
if os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME') or not os.access(BASE_DIR, os.W_OK):
    app.config['UPLOAD_FOLDER'] = os.path.join('/tmp', 'uploads')
else:
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')

try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
except Exception:
    app.config['UPLOAD_FOLDER'] = os.path.join('/tmp', 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)


def resolve_input_image_path(filename):
    """Resolve content or style input image from uploads or bundled sample presets."""
    if not filename:
        return None
    # 1. User uploaded file in upload folder
    p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(p):
        return p
    # 2. Preset images in style_data
    p = os.path.join(BASE_DIR, 'style_data', filename)
    if os.path.exists(p):
        return p
    # 3. Preset images in examples
    p = os.path.join(BASE_DIR, 'examples', filename)
    if os.path.exists(p):
        return p
    # 4. Fallback in static/uploads
    p = os.path.join(BASE_DIR, 'static', 'uploads', filename)
    if os.path.exists(p):
        return p
    return None


# ── ONNX Runtime Inference Sessions ──────────────────────────────────────────
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
    """Resize shortest side to size, maintain aspect ratio, convert to float32 NCHW [0, 1]."""
    w, h = pil_img.size
    if w < h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_w, new_h = int(round(w * size / h)), size
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # HWC [0, 1]
    arr = arr.transpose(2, 0, 1)[np.newaxis]            # NCHW
    return arr


def _adain(content_feat, style_feat, eps=1e-5):
    """Adaptive Instance Normalization in NumPy matching PyTorch implementation exactly."""
    c_mean = content_feat.mean(axis=(2, 3), keepdims=True)
    c_var = np.var(content_feat, axis=(2, 3), keepdims=True)
    c_std = np.sqrt(c_var + eps)

    s_mean = style_feat.mean(axis=(2, 3), keepdims=True)
    s_var = np.var(style_feat, axis=(2, 3), keepdims=True)
    s_std = np.sqrt(s_var + eps)

    normalized = (content_feat - c_mean) / c_std
    return s_std * normalized + s_mean


def run_style_transfer(content_pil, style_pil, alpha=1.0):
    """Executes live AdaIN neural style transfer with ONNX encoder & decoder."""
    content_arr = _preprocess(content_pil)
    style_arr = _preprocess(style_pil)

    # 1. Extract feature maps at relu4_1
    c_feats = _encoder_sess.run(['features'], {'image': content_arr})[0]
    s_feats = _encoder_sess.run(['features'], {'image': style_arr})[0]

    # 2. Perform AdaIN statistical alignment and alpha interpolation
    stylized_feats = _adain(c_feats, s_feats)
    blended_feats = alpha * stylized_feats + (1.0 - alpha) * c_feats

    # 3. Decode features back to RGB image
    out_arr = _decoder_sess.run(['image'], {'features': blended_feats.astype(np.float32)})[0]

    # 4. Post-process to PIL Image
    out_arr = out_arr.squeeze(0).transpose(1, 2, 0)
    out_arr = np.clip(out_arr, 0.0, 1.0)
    out_uint8 = (out_arr * 255.0).astype(np.uint8)
    return Image.fromarray(out_uint8)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        # Handle Content Image: Priority to freshly uploaded file
        if form.content.data and hasattr(form.content.data, 'filename') and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                ext = os.path.splitext(secure_filename(form.content.data.filename))[1].lower()
                content_filename = f"content_{uuid.uuid4().hex[:8]}{ext}"
                dest_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                form.content.data.save(dest_path)
                form.content_path.data = content_filename
        elif form.content_path.data:
            content_filename = form.content_path.data

        # Handle Style Image: Priority to freshly uploaded file
        if form.style.data and hasattr(form.style.data, 'filename') and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                ext = os.path.splitext(secure_filename(form.style.data.filename))[1].lower()
                style_filename = f"style_{uuid.uuid4().hex[:8]}{ext}"
                dest_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                form.style.data.save(dest_path)
                form.style_path.data = style_filename
        elif form.style_path.data:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            c_path = resolve_input_image_path(content_filename)
            s_path = resolve_input_image_path(style_filename)

            if not c_path or not os.path.exists(c_path):
                error = f"Content image '{content_filename}' could not be loaded. Please re-upload."
            elif not s_path or not os.path.exists(s_path):
                error = f"Style image '{style_filename}' could not be loaded. Please re-upload."
            else:
                try:
                    content_img = Image.open(c_path).convert('RGB')
                    style_img = Image.open(s_path).convert('RGB')

                    alpha = float(form.alpha.data) if form.alpha.data is not None else 1.0
                    alpha = max(0.0, min(1.0, alpha))

                    # Perform live neural style transfer
                    stylized_img = run_style_transfer(content_img, style_img, alpha=alpha)

                    # Save to unique filename to prevent any browser caching collisions
                    unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
                    result_filename = f"stylized_{unique_id}.jpg"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    stylized_img.save(result_path, quality=95)

                    result_image = result_filename
                except Exception as e:
                    error = f"Style transfer failed: {str(e)}"
        else:
            if request.method == 'POST':
                if not content_filename:
                    error = "Please upload or select a Content Image."
                elif not style_filename:
                    error = "Please upload or select a Style Image."

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
        cache_buster=int(time.time())
    )


@app.route('/uploads/<filename>')
def send_image(filename):
    # 1. Generated stylized images and uploads from /tmp/uploads
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(upload_path):
        resp = make_response(send_from_directory(app.config['UPLOAD_FOLDER'], filename))
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp

    # 2. Preset images from examples
    ex_path = os.path.join(BASE_DIR, 'examples', filename)
    if os.path.exists(ex_path):
        return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)

    # 3. Preset images from style_data
    st_path = os.path.join(BASE_DIR, 'style_data', filename)
    if os.path.exists(st_path):
        return send_from_directory(os.path.join(BASE_DIR, 'style_data'), filename)

    # 4. Preset images from static/uploads
    fallback_path = os.path.join(BASE_DIR, 'static', 'uploads', filename)
    if os.path.exists(fallback_path):
        return send_from_directory(os.path.join(BASE_DIR, 'static', 'uploads'), filename)

    return "Image not found", 404


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)