

def get_device():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    return torch.device(device=device)


def show_batch(title: str, images, labels, label_map: dict, device=None, net=None, path=None, grad_cam=False, target_layer=None):
    '''
    Plota um batch. Se o modelo não for None, calcula as predições e plota junto. Se path não for None, salva a imagem em path como png. 
    Se grad_cam=True, então o método grad_cam é aplicado, para isso, target_layer tem que ser a última camada convolucional da rede.
    label_map deve ser um dicionário que recebe a predição do modelo e diz o label.
    '''
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    images.to(device)

    prd = None
    if net:
        net.to(device)
        with torch.no_grad():
            images_gpu = images.to(device)
            prd = net(images_gpu).cpu().numpy()
        prd = np.argmax(prd, axis=-1)

    def prepare_img(img):
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2  # Assuming [-1, 1] normalization
        img = np.clip(img, 0, 1)
        return img
    
    grayscale_cams = None
    if grad_cam:
        assert target_layer is not None, "You must give a target_layer when grad_cam=True"
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        cam = GradCAMPlusPlus(model=net, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(int(pred)) for pred in prd]
        grayscale_cams = cam(input_tensor=images, targets=targets)

    n = len(images)
    grid_size = int(np.ceil(np.sqrt(n)))

    fig = plt.figure(figsize=(grid_size * 2.5, grid_size * 2.5))
    fig.suptitle(title, fontsize=16)

    for idx in range(n):
        ax = plt.subplot(grid_size, grid_size, idx + 1)
        img = prepare_img(images[idx])
        
        visualization_img = None
        if grad_cam:
            cam_map = grayscale_cams[idx]
            visualization_img = show_cam_on_image(img, cam_map, use_rgb=True)
        else:
            visualization_img = img

        ax.imshow(visualization_img)
        ax.grid(False)
        ax.axis('off')

        true_label_idx = labels[idx].item() if torch.is_tensor(labels[idx]) else labels[idx]
        true_label = label_map[true_label_idx]
        subtitle = f'True: {true_label}'

        if prd is not None:
            pred_label = label_map[prd[idx]]
            subtitle += f'\nPrd: {pred_label}'

        ax.set_title(subtitle, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if path:
        plt.savefig(path, dpi=300)
    plt.show()
    
    return