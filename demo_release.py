import argparse
import matplotlib.pyplot as plt
import torch
import os
import glob

from colorizers import *
from colorizers.custom_colorizer import eccv16_custom

def load_custom_model(weights_path, use_gpu=False):
    """Load custom trained model"""
    model = eccv16_custom(pretrained=False, weights_path=weights_path)
    if use_gpu:
        model.cuda()
    model.eval()
    return model

def test_small_dataset(model, test_folder, use_gpu=False, num_test=5):
    """Quick test on a few images from the dataset"""
    print(f"🧪 Quick testing on {num_test} images from {test_folder}")
    
    # Find test images
    test_images = sorted(glob.glob(os.path.join(test_folder, "*.jpg")))[:num_test]
    
    if not test_images:
        print("No test images found!")
        return
    
    for img_path in test_images:
        print(f"Testing: {os.path.basename(img_path)}")
        try:
            # Load and process image
            img = load_img(img_path)
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
            
            if use_gpu:
                tens_l_rs = tens_l_rs.cuda()
            
            # Generate colorization
            with torch.no_grad():
                out_img = postprocess_tens(tens_l_orig, model(tens_l_rs).cpu())
            
            # Save result
            output_path = f"test_{os.path.basename(img_path)}"
            plt.imsave(output_path, out_img)
            print(f"   Saved: {output_path}")
            
        except Exception as e:
            print(f"   Error processing {img_path}: {e}")

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
parser.add_argument('--custom_weights', type=str, default='eccv16_myweights.pth', help='path to custom trained weights')
parser.add_argument('--quick_test', action='store_true', help='quick test on dataset images')
parser.add_argument('--test_folder', type=str, default='imgs/train_black', help='folder for quick testing')
parser.add_argument('--num_test', type=int, default=3, help='number of images for quick test')
opt = parser.parse_args()

# load colorizers
try:
    colorizer_eccv16_custom = load_custom_model(opt.custom_weights, opt.use_gpu)
    print("✅ Custom model loaded successfully!")
    
    # Quick test on dataset images if requested
    if opt.quick_test:
        test_small_dataset(colorizer_eccv16_custom, opt.test_folder, opt.use_gpu, opt.num_test)
        exit(0)
        
except Exception as e:
    print(f"❌ Error loading custom model: {e}")
    print("Falling back to pretrained model...")
    colorizer_eccv16_custom = eccv16_custom(pretrained=True).eval()
    if opt.use_gpu:
        colorizer_eccv16_custom.cuda()

colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if opt.use_gpu:
    colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if opt.use_gpu:
    tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))

with torch.no_grad():
    out_img_eccv16_custom = postprocess_tens(tens_l_orig, colorizer_eccv16_custom(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

plt.imsave('%s_eccv16_custom.png' % opt.save_prefix, out_img_eccv16_custom)
plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)

print(f"✅ Results saved:")
print(f"   - Custom model: {opt.save_prefix}_eccv16_custom.png")
print(f"   - SIGGRAPH17: {opt.save_prefix}_siggraph17.png")

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_eccv16_custom)
plt.title('Output (Custom ECCV 16)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()