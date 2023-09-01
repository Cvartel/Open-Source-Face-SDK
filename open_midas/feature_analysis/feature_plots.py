import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_outliers(csv_ds, outliers_bn_numbers, label_of_classes, selected_feature, quantile_values, quantile=None,
                  quantile_range=None, selected_feature_name="mean of channel ", start_layer=0, end_layer=100,
                  label=None,
                  colors={0: 'red', 1: 'navy', 2: 'blue', 3: 'lime', 4: 'yellow', 5: 'forestgreen'},
                  # colors = {0:'green', 1:'red', 2:'orange', 3:'blue', 4:'purple', 5:'magenta'},
                  alphas={0: 0.3, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},
                  labels_name={0: 'real', 1: '2d_mask', 2: '3d_mask', 3: 'photo', 4: 'photo_regions', 5: 'replay', }):
    for i in outliers_bn_numbers:
        fig = plt.figure(figsize=(23, 4))
        ax1 = fig.add_axes([0.10, 0.10, 0.70, 0.85])
        _label = ''

        if quantile_range:
            indexes_chosen_quantille = selected_feature[
                (selected_feature[selected_feature_name + f"_{i}"] > quantile_values[0][i]) &
                (selected_feature[selected_feature_name + f"_{i}"] < quantile_values[1][i])].index.values
        else:
            if quantile == 'high':
                _label = quantile
                indexes_chosen_quantille = selected_feature[
                    selected_feature[selected_feature_name + f"_{i}"] > quantile_values[i]].index.values
            elif quantile == 'low':
                _label = quantile
                indexes_chosen_quantille = selected_feature[
                    selected_feature[selected_feature_name + f"_{i}"] < quantile_values[i]].index.values

        for key, value in label_of_classes.items():
            samples = selected_feature.iloc[indexes_chosen_quantille][
                csv_ds["label"][indexes_chosen_quantille] == value]

            cur = np.mean(samples.values, axis=0)[start_layer:end_layer]
            std = np.std(samples.values, axis=0)[start_layer:end_layer]

            ax1.plot(range(start_layer, end_layer), cur, label=f"{labels_name[value]}", color=colors[value])
            ax1.fill_between(range(start_layer, end_layer), cur + std, cur - std, alpha=alphas[value],
                             color=colors[value])

        if label:
            _label = f"{label} quantile"

        plt.title(f"{_label} sample from bn number  {i}")

        if end_layer - start_layer < 100:
            plt.xticks(range(start_layer, end_layer))
        else:
            plt.xticks(range(start_layer, end_layer, 2))
        plt.legend()
        plt.grid()
        plt.show()


def plot_outliers_or_range(dataset_csv, plot_all_layers=False, start_layer=None, end_layer=None, high_quantile=0.95,
                           low_quantile=0.05,
                           name_of_classes={'real': 0, '2d_mask': 1, '3d_mask': 2, 'photo': 3, 'photo_regions': 4,
                                            'replay': 5, },
                           outliers_bn_numbers=[10], _range=False):
    name_features = ["mean of channel ", "variance of channel", "L2 between mean of channel and bn running mean",
                     "L2 between variance of channel and bn running variance", "mean of channel variance",
                     "variance of channel variance",
                     "mean of channel variance after centering", "variance of channel variance after centering"]

    selected_feature_name = name_features[0]
    selected_feature = dataset_csv.filter(like=selected_feature_name, axis=1)
    high = np.quantile(selected_feature, 0.95, axis=0)
    low = np.quantile(selected_feature, 0.05, axis=0)

    if plot_all_layers:
        start_layer = 0
        end_layer = selected_feature.shape[1]

    if _range:
        for i in outliers_bn_numbers:
            plot_outliers(dataset_csv, [i], name_of_classes, selected_feature, (low, high),
                          label=f"{int(low_quantile * 100)}-{int(high_quantile * 100)}%",
                          quantile_range=True, start_layer=start_layer, end_layer=end_layer)
    else:
        for i in outliers_bn_numbers:
            plot_outliers(dataset_csv, [i], name_of_classes, selected_feature, high, "high", start_layer=start_layer,
                          end_layer=end_layer)
            plot_outliers(dataset_csv, [i], name_of_classes, selected_feature, low, "low", start_layer=start_layer,
                          end_layer=end_layer)


def plot_feature_and_fill_std(name_features, dataset_csv,
                              name_of_classes={'real': 0, '2d_mask': 1, '3d_mask': 2, 'photo': 3, 'photo_regions': 4,
                                               'replay': 5, },
                              plot_all_layers=False,
                              start_layer=None, end_layer=None,
                              colors={0: 'red', 1: 'navy', 2: 'blue', 3: 'lime', 4: 'yellow', 5: 'forestgreen'},
                              # colors = {0:'green', 1:'red', 2:'orange', 3:'blue', 4:'purple', 5:'magenta'},
                              alphas={0: 0.3, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}):
    for i in name_features:
        fig = plt.figure(figsize=(23, 4))
        ax1 = fig.add_axes([0.10, 0.10, 0.70, 0.85])

        selected_feature = dataset_csv.filter(like=i, axis=1)
        if plot_all_layers:
            start_layer = 0
            end_layer = selected_feature.shape[1]

        for key, value in name_of_classes.items():
            samples = selected_feature[dataset_csv["label"] == value]

            cur = np.mean(samples, axis=0)[start_layer:end_layer]
            std = np.std(samples, axis=0)[start_layer:end_layer]

            ax1.plot(range(start_layer, end_layer), cur, label=f"{key}", color=colors[value])
            ax1.fill_between(range(start_layer, end_layer), cur + std, cur - std, alpha=alphas[value],
                             color=colors[value])

        plt.title(f"{i}")
        if end_layer - start_layer < 100:
            plt.xticks(range(start_layer, end_layer))
        else:
            plt.xticks(range(start_layer, end_layer, 2))
        plt.legend()
        plt.grid()
        plt.show()


def print_features(image_tensor, feature_maker,
                   clr_bar_range=(0, 1), clr_bar_label="",
                   ticklist=None, ticklabels=None,
                   color_map='jet', numbers_of_features=[0, 1, 2, 3, 4, 5, 6, 7],
                   custom_clr_map=False, transparency=1.0,
                   log_scale=False, printing_batchnorms=None,
                   name_features=["mean of channel ", "variance of channel",
                                  "L2 between mean of channel and bn running mean",
                                  "L2 between variance of channel and bn running variance", "mean of channel variance",
                                  "variance of channel variance",
                                  "mean of channel variance after centering",
                                  "variance of channel variance after centering"]):
    """ image_tensor : torch.Tensor  'tensor containing images for computing'
    clr_bar_range : tuple  'range values for color bar'
    clr_bar_label : string   'label of color bar'
    ticklist : array[float]  'custom tick values'
    ticklabels : array["string"] or "string"   'custom tick labels'
    numbers_of_features : array[int]  feature numbers we want to visualize
    printing_batchnorms : tuple  (start_bn, end_bn) if None: then all batchnorms
    """
    feature_tensor = feature_maker(image_tensor)
    N = image_tensor.shape[0]
    n_batchnorms = feature_tensor.shape[1] // len(name_features)
    if printing_batchnorms is None:
        start_layer = 0
        end_layer = n_batchnorms - 1
    else:
        start_layer = printing_batchnorms[0]
        end_layer = printing_batchnorms[1]
    cmap = plt.get_cmap(color_map, N)

    for i in numbers_of_features:
        fig = plt.figure(figsize=(23, 4))
        ax1 = fig.add_axes([0.10, 0.10, 0.70, 0.85])
        plt.title(f"{name_features[i]}")
        colors = []
        if custom_clr_map:
            for c in color_map.colors:
                for nn in range(N // len(ticklist)):
                    colors.append(c)

        for j, n in enumerate(feature_tensor[:N, n_batchnorms * i:n_batchnorms * (i + 1)]):
            if custom_clr_map:
                ax1.plot(range(start_layer, end_layer), n[start_layer:end_layer], alpha=transparency, c=colors[j])
            else:
                ax1.plot(range(start_layer, end_layer), n[start_layer:end_layer], alpha=transparency, c=cmap(j))

        plt.xlabel('BN number')
        plt.ylabel('metrics')

        norm = mpl.colors.Normalize(vmin=clr_bar_range[0], vmax=clr_bar_range[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if ticklabels:
            cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ticks=ticklist, orientation='vertical')
            cbar.set_ticklabels(ticklabels)
        else:
            cbar = fig.colorbar(sm, ticks=np.linspace(clr_bar_range[0], clr_bar_range[1], N), orientation='vertical')

        if i == 3 and log_scale:
            ax1.set_yscale('log')
        cbar.set_label(clr_bar_label, loc="center")
        plt.xticks(range(start_layer, end_layer))
        plt.grid()
        plt.show()
