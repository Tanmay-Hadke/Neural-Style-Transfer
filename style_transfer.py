import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# ───────────────────────────────────────────────
# CONFIGURABLE PARAMETERS
# ───────────────────────────────────────────────
content_path = r"input-images\sunflowers-8175248_1280.jpg"
style_path = r"styles\obey.png"
output_path = r"output-images\output.jpg"
max_size = 512  # Increase up to 1024 for better quality

content_weight = 1e5
style_weight = 1e3
tv_weight = 1e-6
steps = 500

# ───────────────────────────────────────────────
# DEVICE
# ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────
# IMAGE LOADER
# ───────────────────────────────────────────────
def load_image(path, max_size=max_size):
    image = Image.open(path).convert('RGB')
    size = max(image.size)
    if size > max_size:
        size = max_size
    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    return torch.clamp(image, 0, 1)

# ───────────────────────────────────────────────
# MODEL PREP
# ───────────────────────────────────────────────
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# ───────────────────────────────────────────────
# LAYERS FOR CONTENT & STYLE
# ───────────────────────────────────────────────
content_layer = 'conv4_2'
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G

def total_variation_loss(img):
    x_diff = img[:, :, :, :-1] - img[:, :, :, 1:]
    y_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))

# ───────────────────────────────────────────────
# LOAD IMAGES
# ───────────────────────────────────────────────
content = load_image(content_path)
style = load_image(style_path, max_size=max_size)
target = content.clone().requires_grad_(True).to(device)

# ───────────────────────────────────────────────
# EXTRACT FEATURES
# ───────────────────────────────────────────────
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# ───────────────────────────────────────────────
# OPTIMIZATION
# ───────────────────────────────────────────────
optimizer = optim.LBFGS([target])
run = [0]

def closure():
    optimizer.zero_grad()
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean((target_features[content_layer] - content_features[content_layer]) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_layers:
        target_feat = target_features[layer]
        target_gram = gram_matrix(target_feat)
        style_gram = style_grams[layer]
        _, d, h, w = target_feat.shape
        style_loss += torch.mean((target_gram - style_gram) ** 2) / (d * h * w)

    # Total Variation Loss
    tv_loss = total_variation_loss(target)

    total_loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
    total_loss.backward()

    if run[0] % 50 == 0:
        print(f"Step {run[0]} | Total Loss: {total_loss.item():.4f}")
    run[0] += 1
    return total_loss

while run[0] <= steps:
    optimizer.step(closure)

# ───────────────────────────────────────────────
# SAVE OUTPUT
# ───────────────────────────────────────────────
final_img = im_convert(target)
plt.imsave(output_path, final_img.permute(1, 2, 0).numpy())
print(f"Stylized image saved to {output_path}")
