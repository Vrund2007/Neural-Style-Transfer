"""
Export NST encoder and decoder to ONNX format.
Run this once locally from the NST_Code directory:
    python export_onnx.py

Produces:
    encoder.onnx  (~75 MB)
    decoder.onnx  (~14 MB)
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils.models import VGGEncoder, Decoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vgg_path     = os.path.join(BASE_DIR, 'vgg_normalised.pth')
decoder_path = os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')

device = torch.device('cpu')

print("Loading encoder...")
encoder = VGGEncoder(vgg_path).to(device)
encoder.eval()

print("Loading decoder...")
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=False))
decoder.eval()


# Wrap encoder so it takes a single tensor input and returns a single tensor
# (ONNX export doesn't support Python bool arguments easily)
class EncoderWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        # Runs encoder in test mode (returns relu4_1 only)
        h1 = self.enc.enc_1(x)
        h2 = self.enc.enc_2(h1)
        h3 = self.enc.enc_3(h2)
        h4 = self.enc.enc_4(h3)
        return h4


encoder_wrapped = EncoderWrapper(encoder)
encoder_wrapped.eval()

# --- Export encoder ---
dummy_image = torch.randn(1, 3, 512, 512)
encoder_out_path = os.path.join(BASE_DIR, 'encoder.onnx')
print(f"Exporting encoder to {encoder_out_path} ...")
with torch.no_grad():
    torch.onnx.export(
        encoder_wrapped,
        dummy_image,
        encoder_out_path,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image':    {0: 'batch', 2: 'height', 3: 'width'},
            'features': {0: 'batch', 2: 'feat_h',  3: 'feat_w'},
        },
        opset_version=11,
        dynamo=False,
    )
print(f"  -> Saved ({os.path.getsize(encoder_out_path)/1024/1024:.1f} MB)")

# --- Export decoder ---
dummy_features = torch.randn(1, 512, 64, 64)
decoder_out_path = os.path.join(BASE_DIR, 'decoder.onnx')
print(f"Exporting decoder to {decoder_out_path} ...")
with torch.no_grad():
    torch.onnx.export(
        decoder,
        dummy_features,
        decoder_out_path,
        input_names=['features'],
        output_names=['image'],
        dynamic_axes={
            'features': {0: 'batch', 2: 'feat_h', 3: 'feat_w'},
            'image':    {0: 'batch', 2: 'height', 3: 'width'},
        },
        opset_version=11,
        dynamo=False,
    )
print(f"  -> Saved ({os.path.getsize(decoder_out_path)/1024/1024:.1f} MB)")
print("\nDone! ONNX models exported successfully.")
