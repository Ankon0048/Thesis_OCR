import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks


def auto_savgol_smooth(profile, polyorder=3, spacing_factor=1.2,
                       plot=False,  # now default False – caller decides if it wants a figure back
                       plot_title="",
                       save_path=None):

    peaks, _ = find_peaks(profile, distance=8)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected to estimate line spacing.")

    # ------------------------------------------------------------------
    # 2) WEIGHTED average spacing
    # ------------------------------------------------------------------
    # raw peak‑to‑peak distances
    diffs = np.diff(peaks)            # e.g. [35, 37, 110, 36, 34]

    # weight each distance inversely with its size:
    #   smaller distance  -> bigger weight
    #   larger distance   -> smaller weight
    # add a tiny epsilon to avoid division by zero
    eps = 1e-9
    weights = 1.0 / (diffs + eps)

    # weighted average
    avg_spacing = int(np.round(np.sum(weights * diffs) / np.sum(weights)))
    # ------------------------------------------------------------------

    # 3) choose window_length (unchanged except it now uses new avg_spacing)
    window_length = int(spacing_factor * avg_spacing)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2)
    window_length = min(window_length,
                        len(profile) - 1 if len(profile) % 2 else len(profile) - 2)

    # 4) filter
    smoothed = savgol_filter(profile, window_length=window_length, polyorder=polyorder)

    # 5) optional plot
    if plot:
        fig = plt.figure(figsize=(14, 5))
        plt.plot(profile, label="Original", color="orange", alpha=0.6)
        plt.plot(smoothed, label=f"Smoothed (window={window_length})", color="blue")
        plt.plot(peaks, profile[peaks], "rx", label="Detected Peaks")
        plt.title(plot_title or "Savitzky-Golay smoothing")
        plt.xlabel("Row Index")
        plt.ylabel("Sum of Pixel Intensities")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        plt.close(fig)

    return smoothed


def calculate_projection_profile_and_crop_lines(image_path, output_dir):
    """Creates two plots: projection profile and comparison plot."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # binarise (Otsu)
    _, binary_image = cv2.threshold(image, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # horizontal projection profile
    horizontal_projection = np.sum(binary_image, axis=1)

    # -------- Savitzky–Golay smoothing + comparison plot ----------
    comp_plot_path = Path(output_dir) / f"{image_path.stem}_comparison_plot.png"
    smoothed = auto_savgol_smooth(horizontal_projection,
                                  plot=True,
                                  plot_title=f"Savitzky-Golay Smoothing\n{image_path.name}",
                                  save_path=comp_plot_path)

    # -------- threshold & line segmentation ----------
    threshold = 0.1 * np.max(smoothed)
    line_ranges, is_in_line, start_row = [], False, 0
    for row, val in enumerate(smoothed):
        if val > threshold and not is_in_line:
            start_row, is_in_line = row, True
        elif val <= threshold and is_in_line:
            line_ranges.append((start_row, row))
            is_in_line = False

    # # -------- projection‑profile plot (unchanged) ----------
    # proj_plot_path = Path(output_dir) / f"{image_path.stem}_projection_profile.png"
    # fig = plt.figure(figsize=(10, 4))
    # plt.plot(smoothed, range(binary_image.shape[0]),
    #          color='b', label='Projection Profile')
    # plt.axvline(x=threshold, color='r', linestyle='--',
    #             label=f'Threshold = {threshold:.2f}')
    # plt.gca().invert_yaxis()
    # plt.title(f"Horizontal Projection Profile\n{image_path.name}")
    # plt.xlabel("Sum of Pixel Intensities")
    # plt.ylabel("Row Index")
    # plt.legend()
    # plt.tight_layout()
    # fig.savefig(proj_plot_path)
    # plt.close(fig)

    # print(f"Saved: {proj_plot_path}")
    # print(f"Saved: {comp_plot_path}")


# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
input_dir = Path(r"E://Mass_Line_Extraction//output_images//handwritten")
output_dir = Path(r"E://Mass_Line_Extraction//Temporary")
output_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for image_file in input_dir.iterdir():
    if image_file.suffix.lower() in image_extensions:
        calculate_projection_profile_and_crop_lines(image_file, output_dir)
