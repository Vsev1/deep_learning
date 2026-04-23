import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from deep_learning import FashionMLP


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionMLP().to(device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.eval()

    print("Model is loaded")

    classes = [
        "T-shirt/top", "Trousers", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    img = "dataset/torch_example_image3.jpeg"

    image_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(img)

    x = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

        # probs = torch.softmax(pred, dim=1)[0]

        predicted_class_index = pred.argmax(dim=1).item()
        predicted_class_name = classes[predicted_class_index]

    print(f"\nImage: {img}")
    print(f"Predicted class: {predicted_class_name}")

    # print("\nAll class probabilities:")
    # for i in range(len(classes)):
    #     print(f"{classes[i]} — {probs[i].item() * 100:.2f}%")

    plt.imshow(image, cmap="gray")
    plt.title(f"Prediction: {predicted_class_name}")
    plt.show()


if __name__ == '__main__':
    main()
