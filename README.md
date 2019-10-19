# Car Damage Detection

Requirements:
- Python 3.7
- PyTorch 1.3
- TorchVision 0.4
- Pillow 6.2

## How to Run on Windows:

1. Make sure you have Python 3.7
2. Install dependencies:
```bash
pip3 install torch===1.3.0 torchvision===0.4.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install pillow
``` 
3. Run app.py with the name of the image you want to process. The model reads files from 
`./data/validation/00-damage/{name_including_extension}`.
```bash
python3 app.py {name_including_extension}
```