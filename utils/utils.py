import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def get_colormap():
    base_colors = plt.cm.tab20b(np.linspace(0, 1, 20))
    return ListedColormap(base_colors[:19])

def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()
    cmap = get_colormap()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for j in range(images.size(0)):
                img = images[j].cpu().permute(1, 2, 0).numpy()
                img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
                img = np.clip(img, 0, 1)

                label = labels[j].cpu().numpy()
                pred = preds[j].cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                axs[0].imshow(img)
                axs[0].set_title("Input Image")
                axs[1].imshow(label, cmap=cmap)
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred, cmap=cmap)
                axs[2].set_title("Prediction")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
