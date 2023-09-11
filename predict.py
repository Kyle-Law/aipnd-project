import argparse
import torch
from torchvision import models
from torch import optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np

# Given load_checkpoint function here...
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    epochs = checkpoint['epochs']
    
    return model, optimizer, epochs
# Given process_image function here...
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Load the image
    pil_image = Image.open(image_path)
    
    # Resize the image
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((10000, 256)) # constrain the height to 256
    else:
        pil_image.thumbnail((256, 10000)) # constrain the width to 256
        
    # Crop the center 224x224 portion of the image
    left_margin = (pil_image.width - 224) / 2
    bottom_margin = (pil_image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert image to numpy array and scale values to range [0, 1]
    np_image = np.array(pil_image) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean) / std
    
    # Reorder dimensions for PyTorch
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
# Given imshow function here...
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
# Given predict function here...
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Process the image
    img = process_image(image_path)
    
    # Convert to PyTorch tensor and add batch dimension
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    img_tensor = img_tensor.unsqueeze(0)  # this is for VGG
    
    # Ensure the tensor is on the same device as the model
    img_tensor = img_tensor.to(device)
    
    # Make predictions
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        logps = model.forward(img_tensor)
    
    # Calculate the class probabilities
    ps = torch.exp(logps)
    
    # Get the topk results
    top_probs, top_indices = ps.topk(topk)
    
    # Convert top_probs to list
    top_probs = top_probs.cpu().numpy().tolist()[0]
    
    # Convert to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices.cpu().numpy()[0]]
    
    return top_probs, top_classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the saved model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default="", help='Path to a JSON file for mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument("--mps", action="store_true", help="Use MPS (Apple Mac) for training")

    args = parser.parse_args()

    # Load model from checkpoint
    model, _, _ = load_checkpoint(args.checkpoint)

    # Determine device
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model.to(device)

    # Make predictions
    probs, classes = predict(args.image_path, model, device, args.top_k)
    
    # Map classes to real names if --category_names is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]
    
    print("Probabilities:", probs)
    print("Classes:", classes)

if __name__ == "__main__":
    main()
