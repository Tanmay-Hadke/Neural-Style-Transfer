
# 🧠 Neural Style Transfer using PyTorch

> A PyTorch-based implementation of Neural Style Transfer, blending the **content of one image** with the **style of another** using a pretrained VGG19 network.

## 🎨 Overview

This project applies artistic styles (like Van Gogh or Monet) to real-world photos using a deep learning approach first introduced in the paper:

> _"A Neural Algorithm of Artistic Style"_  
> **Leon A. Gatys**, Alexander S. Ecker, and Matthias Bethge (2015)

It combines the **content** of one image with the **style** of another using a pretrained **VGG19** model and computes losses based on **content**, **style (Gram matrices)**, and **total variation (smoothness)**.

---

## 🗂️ Repository Structure

```bash
.
├── style_transfer.py      # Main script to run NST
├── utils.py               # Utility functions for image loading & conversion
├── images/                # Folder for style images
├── output.jpg             # Output image (generated result)
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
```

---

## 🖼️ Example Output

| Content Image            | Style Image              | Output Image             |
|--------------------------|--------------------------|--------------------------|
| ![]([samples/content.jpg](https://github.com/Tanmay-Hadke/Neural-Style-Transfer/blob/main/input-images/sunflowers-8175248_1280.jpg)) | ![](samples/style.jpg)   | ![](samples/output.jpg)  |

---

## ⚙️ How It Works

1. **Load Pretrained VGG19** from `torchvision.models`.
2. Extract features from selected layers:
   - Content: `conv4_2`
   - Style: `conv1_1`, `conv2_1`, ..., `conv5_1`
3. Compute:
   - **Content loss**: similarity in high-level features
   - **Style loss**: similarity in **Gram matrices** of features
   - **Total variation loss**: for spatial smoothness
4. Optimize the **target image** using L-BFGS until style and content loss are minimized.

---

## 📥 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

### 2. Create Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🖼️ Set Your Own Content & Style Images

Edit the top of `style_transfer.py`:
```python
content_path = r"path/to/your/content.jpg"
style_path = r"path/to/your/style.jpg"
output_path = "output.jpg"
```

### 🧪 Run the Script
```bash
python style_transfer.py
```

Output will be saved to the path you specified.

---

## 📌 Key Parameters (Editable in `style_transfer.py`)

| Parameter         | Description                                     | Default  |
|------------------|-------------------------------------------------|----------|
| `max_size`        | Max size to resize input images                 | 512 px   |
| `content_weight`  | Importance of structure preservation            | 1e5      |
| `style_weight`    | Importance of texture/style                     | 1e3      |
| `tv_weight`       | Smoothness of generated image                   | 1e-6     |
| `steps`           | Optimization steps (iterations)                 | 500      |

---

## 📚 Background

### What is the Gram Matrix?

The **Gram matrix** captures feature correlations in a convolutional layer. It removes spatial info and retains **texture**, which defines style. Style loss is computed by comparing the Gram matrices of the **style** and **generated** images.

### Loss Functions

```python
Total Loss = Content Loss + Style Loss + TV Loss
```

- **Content Loss**: Compares activations of `conv4_2` layer
- **Style Loss**: Compares Gram matrices at multiple layers
- **Total Variation Loss**: Reduces image noise (smoothing)

---

## 💡 Enhancements Ideas

- Support **Fast Style Transfer** using feed-forward networks
- Style transfer on **videos**
- Add a **web interface** with `Flask` or `Streamlit`
- Use **argparse** for CLI control

---

## 📷 Credits

- Based on ideas from the paper [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576)
- Style images: [WikiArt](https://www.wikiart.org/)
- Pretrained models: [TorchVision](https://pytorch.org/vision/stable/models.html)

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

Thanks to the PyTorch team and the original creators of Neural Style Transfer.  
Built for educational and creative purposes 🎨🖼️

---

## 👨‍💻 Author

**Your Name**  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourname)
