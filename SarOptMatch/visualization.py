import os

import numpy as np
import tensorflow as tf

import tkinter as tk
from PIL import ImageTk, ImageGrab, Image, ImageDraw

from matplotlib import cm


class ImageGui_opt_sar(tk.Tk):
    """
    GUI for visualizing the SAR-opt dataset interactively.
    """
    from PIL import ImageGrab  # 记得加 import

    def __init__(self, master, opt, sar, offset, heatmaps=None, feature_maps=None, dataset_name="dataset"):
        self.dataset_name = dataset_name
        """
        Initialise GUI
        """

        # So we can quit the window from within the functions
        self.master = master

        # Extract the frame so we can draw stuff on it
        frame = tk.Frame(master)

        # Initialise grid
        frame.grid()
        # Start at the first file name
        assert opt.shape[0] == sar.shape[0]
        self.index = 0
        self.n_images = opt.shape[0]

        self.opt = opt
        self.sar = sar
        self.offset = offset
        self.heatmaps = heatmaps
        self.feature_maps = feature_maps

        # Set empty image container
        self.image = None
        self.image2 = None
        self.image3 = None

        if not self.heatmaps is None:
            self.image4 = None

        if not self.feature_maps is None:
            self.image5 = None
            self.image6 = None

            if len(self.feature_maps) == 4:
                # it means we have multiscale feature too, we need to add two more containers
                self.image7 = None
                self.image8 = None

        # Set image panels
        self.image_panel = tk.Label(frame)
        self.image_panel2 = tk.Label(frame)
        self.image_panel3 = tk.Label(frame)

        if not self.heatmaps is None:
            self.image_panel4 = tk.Label(frame)

        if not self.feature_maps is None:
            self.image_panel5 = tk.Label(frame)
            self.image_panel6 = tk.Label(frame)

            if len(self.feature_maps) == 4:
                # it means we have multiscale feature too, we need to add two more containers
                self.image_panel7 = tk.Label(frame)
                self.image_panel8 = tk.Label(frame)

        # Set image container to first image
        self.set_image()

        self.buttons = []

        ### added in version 2
        self.buttons.append(tk.Button(frame, text="prev im 50", width=10, height=2, fg="purple",
                                      command=lambda l=0: self.show_prev_image50()))
        self.buttons.append(tk.Button(frame, text="prev im", width=10, height=2, fg="purple",
                                      command=lambda l=0: self.show_prev_image()))
        self.buttons.append(tk.Button(frame, text="next im", width=10, height=2, fg='purple',
                                      command=lambda l=0: self.show_next_image()))
        self.buttons.append(tk.Button(frame, text="next im 50", width=10, height=2, fg='purple',
                                      command=lambda l=0: self.show_next_image50()))
        # # Save button
        # self.buttons.append(tk.Button(frame, text="SAVE", width=10, height=2, fg='green', command=lambda l=0: self.save()))

        # Add progress label
        progress_string = "%d/%d" % (self.index + 1, self.n_images)
        self.progress_label = tk.Label(frame, text=progress_string, width=10)
        self.progress_label.grid(row=2, column=3, sticky='we')

        # Place buttons in grid
        for ll, button in enumerate(self.buttons):
            button.grid(row=0, column=ll, sticky='we')
            frame.grid_columnconfigure(ll, weight=1)

            # Place the image in grid

        # Optical
        self.image_panel.grid(row=1, column=0, sticky='we')
        # SAR
        self.image_panel2.grid(row=1, column=1, sticky='we')

        # Optical-SAR
        self.image_panel3.grid(row=1, column=2, sticky='we')
        if not self.heatmaps is None:
            # Heatmaps
            self.image_panel4.grid(row=1, column=3, sticky='we')
        if not self.feature_maps is None:
            # Feature map optical (original resolution)
            self.image_panel5.grid(row=2, column=0, sticky='we')
            # Feature map SAR (original resolution)
            self.image_panel6.grid(row=2, column=1, sticky='we')
            if len(self.feature_maps) == 4:
                # Feature map optical (downscaled)
                self.image_panel7.grid(row=3, column=0, sticky='we')
                # Feature map SAR (downscaled)
                self.image_panel8.grid(row=3, column=1, sticky='we')
            # ===== 自动保存所有图像 =====
            print(f"[AutoSave] Saving {self.n_images} images to 'visual_maps/{self.dataset_name}/'")
            save_dir = os.path.join("visual_maps", self.dataset_name)
            os.makedirs(save_dir, exist_ok=True)

            for i in range(self.n_images):
                self.index = i
                self.save(save_dir=save_dir, only_save=True)

            print("[AutoSave] Done.")
            # ✅ 重置 index 回到第一个图像
            self.index = 0
            self.set_image()

    # ******************************************************************************

    def numpy2PIL(self, data: np.array):
        """ Convert a np.array into a imagePIL
        'data' should be (N, H, W, Chs) """

        if len(data.shape) == 3:
            imagePIL = Image.fromarray(data[self.index, :, :])

        elif len(data.shape) == 4:
            if data.shape[-1] == 1:
                imagePIL = Image.fromarray(data[self.index, :, :, 0])
            else:
                imagePIL = Image.fromarray(data[self.index, :, :, :])

        else:
            raise ("Array has dimensions: " + str(data.shape) + ". Check dimensions")
        return imagePIL

    def save(self, save_dir="visual_maps", only_save=False):
        """
        Save the images and visualizations into the 'visual_maps' directory (or a specific subdirectory).
        """
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

        # Save opt superimposed with SAR
        imagePIL_opt = self.numpy2PIL(self.opt)
        imagePIL_sar = self.numpy2PIL(self.sar)

        row_true, col_true = np.unravel_index(np.argmax(self.offset[self.index, :, :]), shape=(65, 65))
        imagePIL_opt.paste(imagePIL_sar, (col_true, row_true))
        imagePIL_opt.save(os.path.join(save_dir, f"{self.index}_SAR_opt.png"))

        if not self.heatmaps is None:
            heatmap = self.heatmaps[self.index, :, :]
            heatmap_PIL = Image.fromarray(cm.viridis(heatmap, bytes=True))

            # Draw true and predicted matches
            draw = ImageDraw.Draw(heatmap_PIL)
            draw.ellipse((col_true, row_true, (col_true + 2), (row_true + 2)), fill=(255, 0, 0), outline=(0, 0, 0))

            # 预测点也画上
            row_pred, col_pred = np.unravel_index(np.argmax(heatmap), shape=(65, 65))
            draw.ellipse((col_pred, row_pred, (col_pred + 2), (row_pred + 2)), fill=(255, 0, 255), outline=(0, 0, 0))

            heatmap_PIL.save(os.path.join(save_dir, f"{self.index}_heatmap.png"))

        if not self.feature_maps is None:
            psi_opt_o = self.feature_maps[0][self.index, :, :]
            psi_opt_o_PIL = Image.fromarray(cm.jet(psi_opt_o, bytes=True))
            psi_opt_o_PIL.save(os.path.join(save_dir, f"{self.index}_psi_opt_o.png"))

            psi_sar_o = self.feature_maps[1][self.index, :, :]
            psi_sar_o_PIL = Image.fromarray(cm.jet(psi_sar_o, bytes=True))
            psi_sar_o_PIL.save(os.path.join(save_dir, f"{self.index}_psi_sar_o.png"))

            if len(self.feature_maps) == 4:
                psi_opt_d = self.feature_maps[2][self.index, :, :]
                psi_opt_d_PIL = Image.fromarray(cm.jet(psi_opt_d, bytes=True))
                psi_opt_d_PIL.save(os.path.join(save_dir, f"{self.index}_psi_opt_d.png"))

                psi_sar_d = self.feature_maps[3][self.index, :, :]
                psi_sar_d_PIL = Image.fromarray(cm.jet(psi_sar_d, bytes=True))
                psi_sar_d_PIL.save(os.path.join(save_dir, f"{self.index}_psi_sar_d.png"))

    def set_image(self):
        """
        Helper function which sets a new image in the image view
        """

        # 1-Optical:
        imagePIL_opt = self.numpy2PIL(self.opt).resize((256, 256), Image.NEAREST)
        self.image = ImageTk.PhotoImage(imagePIL_opt, master=self.master)
        self.image_panel.configure(image=self.image)

        # 2-SAR: Grayscale
        imagePIL_sar = self.numpy2PIL(self.sar).resize((192, 192), Image.NEAREST)
        self.image2 = ImageTk.PhotoImage(imagePIL_sar, master=self.master)
        self.image_panel2.configure(image=self.image2)

        # 3-Optical super-imposed with SAR
        row_true, col_true = np.unravel_index(np.argmax(self.offset[self.index, :, :]), shape=(65, 65))
        imagePIL_opt.paste(imagePIL_sar, (col_true, row_true))
        self.image3 = ImageTk.PhotoImage(imagePIL_opt, master=self.master)
        self.image_panel3.configure(image=self.image3)

        # 4-Heamaps
        if not self.heatmaps is None:
            heatmap = self.heatmaps[self.index, :, :]
            row_pred, col_pred = np.unravel_index(np.argmax(heatmap), shape=(65, 65))
            heatmapPIL = Image.fromarray(cm.viridis(heatmap, bytes=True)).resize((65 * 2, 65 * 2), Image.NEAREST)

            # Draw true match
            draw = ImageDraw.Draw(heatmapPIL)
            draw.ellipse((col_true * 2, row_true * 2, (col_true + 2) * 2, (row_true + 2) * 2), fill=(255, 0, 0),
                         outline=(0, 0, 0))

            # Draw predicted match
            draw.ellipse((col_pred * 2, row_pred * 2, (col_pred + 2) * 2, (row_pred + 2) * 2), fill=(255, 0, 255),
                         outline=(0, 0, 0))

            self.image4 = ImageTk.PhotoImage(heatmapPIL, master=self.master)
            self.image_panel4.configure(image=self.image4)

        if not self.feature_maps is None:

            # Feature map optical (original resolution)
            psi_opt_o = self.feature_maps[0][self.index, :, :]
            psi_opt_o_PIL = Image.fromarray(cm.jet(psi_opt_o, bytes=True)).resize((256, 256), Image.NEAREST)
            self.image5 = ImageTk.PhotoImage(psi_opt_o_PIL, master=self.master)
            self.image_panel5.configure(image=self.image5)

            # Feature map SAR (original resolution)
            psi_sar_o = self.feature_maps[1][self.index, :, :]
            psi_sar_o_PIL = Image.fromarray(cm.jet(psi_sar_o, bytes=True)).resize((192, 192), Image.NEAREST)
            self.image6 = ImageTk.PhotoImage(psi_sar_o_PIL, master=self.master)
            self.image_panel6.configure(image=self.image6)

            if len(self.feature_maps) == 4:
                # Feature map optical (downscaled)
                psi_opt_d = self.feature_maps[2][self.index, :, :]
                psi_opt_d_PIL = Image.fromarray(cm.jet(psi_opt_d, bytes=True)).resize((256, 256), Image.NEAREST)
                self.image7 = ImageTk.PhotoImage(psi_opt_d_PIL, master=self.master)
                self.image_panel7.configure(image=self.image7)

                # Feature map SAR (downscaled)
                psi_sar_d = self.feature_maps[3][self.index, :, :]
                psi_sar_d_PIL = Image.fromarray(cm.jet(psi_sar_d, bytes=True)).resize((192, 192), Image.NEAREST)
                self.image8 = ImageTk.PhotoImage(psi_sar_d_PIL, master=self.master)
                self.image_panel8.configure(image=self.image8)

    def show_prev_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index -= 1
        progress_string = "%d/%d" % (self.index + 1, self.n_images)
        self.progress_label.configure(text=progress_string)

        if self.index >= 0:
            self.set_image()
        else:
            #   self.master.quit()
            self.master.destroy()

    def show_prev_image50(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index -= 50
        progress_string = "%d/%d" % (self.index + 1, self.n_images)
        self.progress_label.configure(text=progress_string)

        if self.index >= 0:
            self.set_image()
        else:
            #   self.master.quit()
            self.master.destroy()

    def show_next_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 1
        progress_string = "%d/%d" % (self.index + 1, self.n_images)
        self.progress_label.configure(text=progress_string)

        if self.index < self.n_images:
            self.set_image()
        else:
            #            self.master.quit()
            self.master.destroy()

    def show_next_image50(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 50
        progress_string = "%d/%d" % (self.index + 1, self.n_images)
        self.progress_label.configure(text=progress_string)

        if self.index < self.n_images:
            self.set_image()
        else:
            #            self.master.quit()
            self.master.destroy()


def visualize_dataset_with_GUI(dataset: tf.data.Dataset, heatmaps: np.array = None, feature_maps=None,
                               dataset_name="dataset"):
    """
    Visualize a dataset with a GUI:
        opt_image | sAR_image | SAR into opt (correct pos) | similarity heatmap (if provided)
    """

    def process_feature_map(feature_map):
        # Step 1: Average along the last axis
        feature_map = np.mean(feature_map, axis=-1)

        # Step 2: Normalize to mean=0 and std=1, with std protection
        mean = np.mean(feature_map, axis=0) # Corrected: should be over all samples if feature_map is (N,H,W)
        std = np.std(feature_map, axis=0)   # Corrected: same as above
        # If feature_map is already (H,W) after mean, then axis=0 is wrong.
        # Assuming process_feature_map is called on a single feature map (H,W,C) or (N,H,W,C)
        # If called with (N,H,W,C), then mean and std should be calculated per feature map or globally.
        # The original code implies it's called with a list of (N,H,W,C) arrays.
        # Let's assume feature_map input to this function is (N, H, W) after np.mean(axis=-1)
        # Then normalization should be careful.
        # For simplicity, let's assume the original normalization logic was intended per-image if N > 1
        # or globally if N=1. The current axis=0 is problematic for (N,H,W)

        # Re-evaluating normalization for (N,H,W) input after averaging channels
        if feature_map.ndim == 3: # (N, H, W)
            mean_val = np.mean(feature_map, axis=(1, 2), keepdims=True)
            std_val = np.std(feature_map, axis=(1, 2), keepdims=True)
        elif feature_map.ndim == 2: # (H, W)
            mean_val = np.mean(feature_map)
            std_val = np.std(feature_map)
        else: # Should not happen if input is from model
            mean_val = 0
            std_val = 1

        std_val[std_val == 0] = 1e-7  # 防止除以零
        feature_map_norm = (feature_map - mean_val) / std_val

        # Step 3: Rescale to [0, 255], with range protection
        # This should also be done carefully for (N,H,W)
        if feature_map_norm.ndim == 3: # (N, H, W)
            a_min = feature_map_norm.min(axis=(1, 2), keepdims=True)
            a_max = feature_map_norm.max(axis=(1, 2), keepdims=True)
        elif feature_map_norm.ndim == 2: # (H, W)
            a_min = feature_map_norm.min()
            a_max = feature_map_norm.max()
        else: # Should not happen
            a_min = 0
            a_max = 1
            if np.isscalar(feature_map_norm): # if somehow it became scalar
                 if feature_map_norm == 0: a_max = 0 # avoid division by zero if min=max=0
                 elif feature_map_norm > 0: a_min = 0; a_max = feature_map_norm
                 else: a_min = feature_map_norm; a_max = 0


        scale_range = a_max - a_min
        # Handle cases where scale_range might be an array (for per-image normalization)
        if isinstance(scale_range, np.ndarray):
            scale_range[scale_range == 0] = 1e-7
        elif scale_range == 0: # scalar case
            scale_range = 1e-7

        feature_map_scaled = (((feature_map_norm - a_min) / scale_range) * 255).astype('uint8')
        return feature_map_scaled

    dataset = dataset.unbatch() # Each element is ((opt_im, sar_im), (mask, original_offsets))
    opt_list = []
    sar_list = []
    true_mask_list = [] # This will store the actual (65,65) mask arrays

    # Iterate through the tf.dataset
    for element in dataset.as_numpy_iterator():
        inputs_tuple = element[0]    # This is (opt_im, sar_im)
        targets_tuple = element[1]   # This should be (mask_array, original_offsets_array)

        opt_list.append(inputs_tuple[0])
        sar_list.append(inputs_tuple[1])

        # Extract the mask from targets_tuple
        # We expect targets_tuple[0] to be the (65,65) mask array
        if isinstance(targets_tuple, tuple) and len(targets_tuple) >= 1 and \
           hasattr(targets_tuple[0], 'shape') and targets_tuple[0].shape == (65, 65):
            true_mask_list.append(targets_tuple[0])
        elif hasattr(targets_tuple, 'shape') and targets_tuple.shape == (65,65):
            # Fallback for cases where targets_tuple might directly be the mask (old behavior)
            print("Warning: Dataset's target part is directly a mask, not a (mask, original_offsets) tuple. Assuming it's the (65,65) mask.")
            true_mask_list.append(targets_tuple)
        else:
            print(f"Error: Could not extract a valid (65,65) mask from dataset's target part. Target type: {type(targets_tuple)}")
            if hasattr(targets_tuple, 'shape'):
                print(f"Target shape: {targets_tuple.shape}")
            elif isinstance(targets_tuple, tuple) and len(targets_tuple) > 0 and hasattr(targets_tuple[0], 'shape'):
                 print(f"Shape of first element in target tuple: {targets_tuple[0].shape}")
            # Optionally, skip this problematic sample or raise an error
            continue # Skipping this sample

    if not opt_list: # If no valid samples were processed
        print("Error: No valid samples could be processed from the dataset for visualization.")
        return

    opt_np = (np.array(opt_list) * 255).astype('uint8')
    sar_np = (np.array(sar_list) * 255).astype('uint8')

    # Convert true_mask_list to a single NumPy array of shape (N, 65, 65)
    try:
        if not true_mask_list: # Check if the list is empty
            print("Error: true_mask_list is empty. Cannot create mask array for GUI.")
            return
        true_mask_np = np.array(true_mask_list)
        # Ensure the resulting array has the expected dimensions
        if true_mask_np.ndim != 3 or true_mask_np.shape[1:] != (65, 65):
            raise ValueError(f"Constructed true_mask_np has incorrect shape: {true_mask_np.shape}. Expected (N, 65, 65).")
    except ValueError as e:
        print(f"Error converting true_mask_list to NumPy array: {e}")
        print("Debug: Shapes of items in true_mask_list:")
        # for i, item_mask in enumerate(true_mask_list):
        #     if hasattr(item_mask, 'shape'):
        #         print(f"  Item {i} shape: {item_mask.shape}")
        #     else:
        #         print(f"  Item {i} type: {type(item_mask)} (no shape attribute)")
        return

    # 'offset' parameter for ImageGui_opt_sar should now be true_mask_np

    if heatmaps is not None:
        # Process heatmaps for visualization: scale so that each heatmap is to [0,255]
        a_min_h = heatmaps.min(axis=(1, 2), keepdims=True)
        a_max_h = heatmaps.max(axis=(1, 2), keepdims=True)
        scale_range_h = a_max_h - a_min_h
        scale_range_h[scale_range_h == 0] = 1e-7 # Avoid division by zero
        heatmaps_processed = (((heatmaps - a_min_h) / scale_range_h) * 255).astype('uint8')
    else:
        heatmaps_processed = None


    if feature_maps is not None:
        # Process feature maps for visualization
        try:
            feature_maps_processed = list(map(process_feature_map, feature_maps))
        except Exception as e:
            print(f"Error processing feature maps: {e}. Feature maps will not be shown.")
            feature_maps_processed = None
    else:
        feature_maps_processed = None


    # Run GUI
    root = tk.Tk()
    root.withdraw()  # ⚡ 隐藏
    # Pass the correctly structured true_mask_np as the 'offset' argument
    GUI = ImageGui_opt_sar(root, opt_np, sar_np, true_mask_np, heatmaps_processed, feature_maps_processed, dataset_name)
    root.destroy()  # ⚡ 直接销毁，不开窗口



import matplotlib.pyplot as plt


def plot_loss_curve(history, dataset_name="dataset", save_dir="loss_curves"):
    """
    Plot and save the training and validation loss curves.

    Args:
        history: History object returned by model.fit()
        dataset_name: str, name of the dataset (for naming the file)
        save_dir: str, directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Loss Curve - {dataset_name}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{dataset_name}_loss_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Loss曲线已保存: {save_path}")





































