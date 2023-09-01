import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from feature_plots import print_features
from torchvision import transforms


def draw_grid(image, grid_brightness, visualise=False, vis_scale=False):
    if vis_scale:
        print(grid_brightness)
    _image = np.moveaxis(image.numpy(), 0, -1) / 2 + 0.5
    new_image = np.zeros(_image.shape)
    new_image[16::16, ::, :] = grid_brightness
    new_image[::, 16::16, :] = grid_brightness
    if visualise:
        plt_image = np.clip((new_image + _image), 0, 1)
        plt.imshow(plt_image)
        plt.show()
    new_image = (np.clip((new_image + _image), 0, 1) * 2) - 1
    new_image = np.moveaxis(new_image, -1, 0)
    return torch.Tensor(new_image)


def rotate_image(image, rotate_scale, visualise=False, vis_scale=False):
    if vis_scale:
        print(rotate_scale)
    img = image
    rotated_img = transforms.functional.rotate(img, rotate_scale)
    if visualise:
        vis = np.moveaxis(rotated_img.numpy(), 0, -1)
        vis = vis / 2 + 0.5
        plt.imshow(vis)
        plt.show()
    return rotated_img


def blur_image(image, blur_scale, visualise=False, vis_scale=False):
    if vis_scale:
        print(blur_scale)
    img = image
    blur_img = transforms.GaussianBlur(kernel_size=(35, 35), sigma=blur_scale)(img)
    if visualise:
        vis = np.moveaxis(blur_img.numpy(), 0, -1)
        vis = vis / 2 + 0.5
        plt.imshow(vis)
        plt.show()
    return blur_img


def jit_of_color(image, brightness=None, contrast=None, saturation=None, hue=None, visualise=False, vis_scale=False):
    if vis_scale:
        print(brightness)
    out_image = image
    if brightness is not None:
        out_image = (transforms.functional.adjust_brightness(out_image, brightness) - 0.5) * 2
    else:
        out_image = out_image

    if visualise:
        vis = np.moveaxis(out_image.numpy(), 0, -1)
        vis = vis / 2 + 0.5
        plt.imshow(vis)
        plt.show()
    return out_image


def show_image(image):
    vis = np.moveaxis(image.numpy(), 0, -1)
    vis = vis / 2 + 0.5
    plt.imshow(vis)
    plt.show()


def make_multiclass_tensor(dataset_s, african_ds, pets_ds, visualise=False):
    """ *args in this case are the names of datasets """
    image_tensor = []
    number_of_img = 5
    for i in range(number_of_img):
        # image_tensor.append(dataset_s.get_sample_by_type('real', i*6)[0])
        image_tensor.append(dataset_s[i * 5])
        if visualise:
            # show_image(dataset_s.get_sample_by_type('real', i*6)[0])
            show_image(dataset_s[i * 5])
    for i in range(number_of_img):
        image_tensor.append(african_ds[i])
        if visualise:
            show_image(african_ds[i])
    for i in range(number_of_img):
        image_tensor.append(pets_ds[i])
        if visualise:
            show_image(pets_ds[i])
    return torch.stack(image_tensor)


def _print_images(images, rows=5, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    for i in range(rows):
        for j in range(cols):
            ax = axes[i][j]

            img = images[5 * i + j].permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            img = (img * 255).astype(np.uint8)
            ax.imshow(img)


def brightness_augmentation(orig_image, feature_maker, print_images=False):
    images_jutted_color = []
    for i in range(0, 30, 2):
        images_jutted_color.append(
            jit_of_color(orig_image * 0.5 + 0.5, brightness=i / 10, visualise=False, vis_scale=False))
    images_jutted_color = torch.stack(images_jutted_color)
    if print_images:
        _print_images(images_jutted_color, rows=3, cols=5)
    print_features(images_jutted_color, feature_maker, clr_bar_range=(0.0, 2.8), clr_bar_label="brightness",
                   ticklist=[0.0, 1.0, 2.8], ticklabels=["Brightness lower", "Original image", "Brightness higher"],
                   log_scale=True,
                   color_map='jet', numbers_of_features=[0, 1, 2, 3, 4, 5, 6, 7], printing_batchnorms=None)


def rotation_augmentation(orig_image, feature_maker, print_images=False):
    images_with_rotation = []
    for i in range(0, 360, 10):
        images_with_rotation.append(rotate_image(orig_image, i, visualise=False, vis_scale=False))
    images_with_rotation = torch.stack(images_with_rotation)
    if print_images:
        _print_images(images_with_rotation[::2], rows=3, cols=5)
    print_features(images_with_rotation, feature_maker, clr_bar_range=(0, 350), clr_bar_label="Angle rotation",
                   log_scale=True,
                   color_map='gnuplot', transparency=1.0, numbers_of_features=[0, 1, 2, 3, 4, 5, 6, 7],
                   printing_batchnorms=None)


def blur_augmentation(orig_image, feature_maker, print_images=False):
    images_with_blur = []
    for i in range(1, 250, 5):
        images_with_blur.append(blur_image(orig_image, i / 10, vis_scale=False))
    images_with_blur = torch.stack(images_with_blur)
    if print_images:
        _print_images(images_with_blur[::5], rows=2, cols=5)

    print_features(images_with_blur, feature_maker, clr_bar_range=(0.0, 24.6), clr_bar_label="blur scale",
                   ticklist=[0.1, 24.6], ticklabels=["Original image", "Max blur"], color_map='gnuplot',
                   log_scale=True, numbers_of_features=[0, 1, 2, 3, 4, 5, 6, 7], printing_batchnorms=None)


def grid_augmentation(orig_image, feature_maker, print_images=False):
    images_with_grid = []
    n_of_grid_image = 20
    for i in range(0, n_of_grid_image):
        images_with_grid.append(
            draw_grid(orig_image, ((i - n_of_grid_image / 2) / n_of_grid_image) * 2, vis_scale=False))

    images_with_grid = torch.stack(images_with_grid)
    if print_images:
        _print_images(images_with_grid, rows=4, cols=5)

    print_features(images_with_grid, feature_maker, clr_bar_range=(-0.5, 0.45), clr_bar_label="grid brightness",
                   ticklist=[-0.5, 0, 0.45], ticklabels=["Black grid", "Original image/no grid", "White grid"],
                   log_scale=True,
                   numbers_of_features=[0, 1, 2, 3, 4, 5, 6, 7], printing_batchnorms=None)


def multiclass_statistic(seleba_ds, african_ds, pets_ds, feature_maker, print_images=False):
    dif_class_image = make_multiclass_tensor(seleba_ds, african_ds, pets_ds, visualise=False)

    if print_images:
        label_names = ["White human", "African human", "Animal"]
        rows = len(label_names)
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
        for i in range(rows):
            fig.text(0.08, (0.8 / rows * (i + 1)) - 0.01, f'{label_names[rows - i - 1]}', ha='center')
            for j in range(cols):
                ax = axes[i][j]

                img = dif_class_image[5 * i + j].permute(1, 2, 0).numpy()
                img = (img + 1) / 2
                img = (img * 255).astype(np.uint8)
                ax.imshow(img)

    color_map = mpl.colors.ListedColormap(['blue', 'green', 'red'])
    print_features(dif_class_image, feature_maker, clr_bar_range=(0.0, 15.0), clr_bar_label="image class",
                   color_map=color_map, custom_clr_map=True,
                   ticklist=[0.0, 5.0, 10.0], ticklabels=["White human", "African human", "Animal"], log_scale=True,
                   numbers_of_features=[0, 1, 2, 3, 4, 5, 6, 7], printing_batchnorms=None)
