import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import pathlib  # 导入 pathlib


# (create_checkerboard_pattern 函数保持不变)
def create_checkerboard_pattern(img1, img2, block_size=32):
    if img1.shape != img2.shape:
        print(f"Error: Images for checkerboard must have the same shape. Got {img1.shape} and {img2.shape}")
        try:
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            print(f"Warning: Resized img2 from {img2.shape} to {img1.shape} to match img1 for checkerboard.")
            img2 = img2_resized
        except Exception as e_resize:
            print(f"Error resizing img2: {e_resize}. Cannot create checkerboard.")
            return None

    h, w = img1.shape[:2]
    checkerboard = np.zeros_like(img1)

    for r_idx in range(0, h, block_size):
        for c_idx in range(0, w, block_size):
            use_img1 = ((r_idx // block_size) % 2 == (c_idx // block_size) % 2)
            r_end = min(r_idx + block_size, h)
            c_end = min(c_idx + block_size, w)

            if use_img1:
                checkerboard[r_idx:r_end, c_idx:c_end] = img1[r_idx:r_end, c_idx:c_end]
            else:
                checkerboard[r_idx:r_end, c_idx:c_end] = img2[r_idx:r_end, c_idx:c_end]
    return checkerboard


def visualize_large_images_with_net_ransac_shift(
        original_optical_large_img_path,
        original_sar_large_img_path,
        M_affine_global_ransac,
        artificial_offset_xy,
        output_visualization_path="large_image_net_shift_checkerboard.png",  # 通常是相对路径
        checkerboard_block_size=128
):
    opt_large_img_orig = cv2.imread(original_optical_large_img_path)
    sar_large_img_orig = cv2.imread(original_sar_large_img_path)

    if opt_large_img_orig is None:
        print(f"错误: 无法从 {original_optical_large_img_path} 加载原始光学图像")
        return
    if sar_large_img_orig is None:
        print(f"错误: 无法从 {original_sar_large_img_path} 加载原始SAR图像")
        return

    if len(opt_large_img_orig.shape) == 2:
        opt_large_img_orig = cv2.cvtColor(opt_large_img_orig, cv2.COLOR_GRAY2BGR)
    if len(sar_large_img_orig.shape) == 2:
        sar_large_img_orig = cv2.cvtColor(sar_large_img_orig, cv2.COLOR_GRAY2BGR)

    if opt_large_img_orig.shape != sar_large_img_orig.shape:
        print(f"警告: 原始大图像形状不同。光学图像: {opt_large_img_orig.shape}, SAR图像: {sar_large_img_orig.shape}.")
        print("正在调整SAR图像大小以匹配光学图像进行可视化。")
        try:
            sar_large_img_orig = cv2.resize(sar_large_img_orig,
                                            (opt_large_img_orig.shape[1], opt_large_img_orig.shape[0]))
        except Exception as e_resize_sar:
            print(f"错误: 调整SAR图像大小时失败: {e_resize_sar}")
            return

    canvas_height, canvas_width = opt_large_img_orig.shape[:2]
    net_shift_tx = 0.0
    net_shift_ty = 0.0

    if M_affine_global_ransac is None:
        print("警告: 全局RANSAC仿射矩阵 M_affine_global_ransac 为 None。无法计算净偏移。")
    else:
        tx_RANSAC = M_affine_global_ransac[0, 2]
        ty_RANSAC = M_affine_global_ransac[1, 2]
        dx_artificial, dy_artificial = artificial_offset_xy
        net_shift_tx = dx_artificial + tx_RANSAC
        net_shift_ty = dy_artificial + ty_RANSAC
        print(f"人为偏移 (dx, dy): ({dx_artificial}, {dy_artificial})")
        print(f"RANSAC平移 (tx, ty): ({tx_RANSAC:.3f}, {ty_RANSAC:.3f})")
        print(f"应用于原始SAR的净偏移 (dx, dy): ({net_shift_tx:.3f}, {net_shift_ty:.3f})")

    translation_matrix_net = np.float32([[1, 0, net_shift_tx], [0, 1, net_shift_ty]])
    sar_large_img_net_shifted = cv2.warpAffine(
        sar_large_img_orig,
        translation_matrix_net,
        (canvas_width, canvas_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    print("正在创建原始光学图像和净偏移后SAR图像的棋盘格可视化...")
    checkerboard_image = create_checkerboard_pattern(
        opt_large_img_orig,
        sar_large_img_net_shifted,
        block_size=checkerboard_block_size
    )

    if checkerboard_image is None:
        print("创建棋盘格图像失败。")
        return

    # 主图像保存 (使用 pathlib，输入通常是相对路径)
    try:
        main_output_path_obj_relative = pathlib.Path(output_visualization_path)
        # 确保其父目录存在 (相对于CWD)
        main_output_path_obj_relative.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(main_output_path_obj_relative), checkerboard_image)
        print(
            f"大图像棋盘格可视化已保存到: {main_output_path_obj_relative} (绝对路径: {main_output_path_obj_relative.resolve()})")
    except Exception as e:
        print(f"保存棋盘格图像 '{output_visualization_path}' 时出错: {e}")
        traceback.print_exc()

    # --- Matplotlib Plot Saving Section ---
    plot_save_path_str_for_error_msg = ""
    try:
        fig = plt.figure(figsize=(12, 12 * (canvas_height / canvas_width if canvas_width > 0 else 1)))
        plt.imshow(cv2.cvtColor(checkerboard_image, cv2.COLOR_BGR2RGB))
        title_str = (f"棋盘格: 原始光学 vs 净效应偏移后的原始SAR\n"
                     f"净偏移 (dx, dy) = ({net_shift_tx:.2f}, {net_shift_ty:.2f}) 像素")
        plt.title(title_str)
        plt.axis('off')

        # --- Path Generation using pathlib (保持相对路径如果输入是相对的) ---
        # output_visualization_path (例如 "subdir/image.png")
        input_path_obj_for_plot = pathlib.Path(output_visualization_path)

        plot_filename_stem = input_path_obj_for_plot.stem
        final_plot_filename = f"{plot_filename_stem}_plot{input_path_obj_for_plot.suffix}"

        # 构建相对的 plot 路径
        # 如果 input_path_obj_for_plot.parent 是 '.', 表示它就在CWD中
        # 否则，它有一个父目录 (如 "subdir")
        if input_path_obj_for_plot.parent != pathlib.Path('.'):
            plot_save_path_obj_relative = input_path_obj_for_plot.parent / final_plot_filename
        else:
            plot_save_path_obj_relative = pathlib.Path(final_plot_filename)

        plot_save_path_str_for_error_msg = str(plot_save_path_obj_relative)
        # --- End Path Generation (relative) ---

        # 确保其父目录存在 (相对于CWD)
        plot_save_path_obj_relative.parent.mkdir(parents=True, exist_ok=True)

        print(f"尝试使用 pathlib (相对路径策略) 保存 Matplotlib 图像到: {plot_save_path_obj_relative}")
        print(f"  (其解析后的绝对路径将是: {plot_save_path_obj_relative.resolve()})")
        print(f"  repr(str(plot_save_path_obj_relative)): {repr(str(plot_save_path_obj_relative))}")

        plt.savefig(plot_save_path_obj_relative)  # 传递相对路径的 Path 对象
        print(
            f"Matplotlib图像也已保存到: {plot_save_path_obj_relative} (绝对路径: {plot_save_path_obj_relative.resolve()})")

    except Exception as e_plt:
        print(
            f"使用Matplotlib保存或显示图像到 '{plot_save_path_str_for_error_msg}' (解析后为 '{pathlib.Path(plot_save_path_str_for_error_msg).resolve()}') 时出错: {e_plt}")
        traceback.print_exc()
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

