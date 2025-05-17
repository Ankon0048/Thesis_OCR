import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

def auto_savgol_smooth(profile, polyorder=2, spacing_factor=0.5, plot=True):
    # Step 1: Detect peaks
    peaks, _ = find_peaks(profile, distance=10)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected to estimate line spacing.")
    
    # Step 2: Estimate average spacing between peaks
    peak_diffs = np.diff(peaks)
    avg_spacing = int(np.mean(peak_diffs))
    
    # Step 3: Calculate window length
    window_length = int(spacing_factor * avg_spacing)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2)
    window_length = min(window_length, len(profile) - 1 if len(profile) % 2 == 1 else len(profile) - 2)

    # Step 4: Apply Savitzky-Golay filter
    smoothed = savgol_filter(profile, window_length=window_length, polyorder=polyorder)

    # Step 5: Optional visualization
    if plot:
        plt.figure(figsize=(14, 5))
        plt.plot(profile, label="Original", color='gray', alpha=0.6)
        plt.plot(smoothed, label=f"Smoothed (window={window_length})", color='blue')
        plt.plot(peaks, profile[peaks], 'rx', label="Detected Peaks")
        plt.title("Savitzky-Golay Smoothing with Auto Window Length")
        plt.xlabel("Row Index")
        plt.ylabel("Sum of Pixel Intensities")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return smoothed


def calculate_projection_profile_and_crop_lines_with_lines(image_path, output_dir):
    np.set_printoptions(threshold=np.inf) 
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Binarize using Otsu's threshold
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection profile
    horizontal_projection = np.sum(binary_image, axis=1)

    smoothed = auto_savgol_smooth(horizontal_projection)

    # Normalize and find line segments
    threshold = 0.1 * np.max(smoothed)

    line_ranges = []
    is_in_line = False
    start_row = 0
    for row, value in enumerate(smoothed):
        if value > threshold:
            if not is_in_line:
                start_row = row
                is_in_line = True
        else:
            if is_in_line:
                end_row = row
                line_ranges.append((start_row, end_row))
                is_in_line = False

    # Plot and save projection profile
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, range(binary_image.shape[0]), color='b', label='Projection Profile')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.gca().invert_yaxis()
    plt.title(f"Horizontal Projection Profile")
    plt.xlabel("Sum of Pixel Intensities")
    plt.ylabel("Row Index")
    plt.legend()

    output_path = Path(output_dir) / "_projection_profile.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

    if(len(line_ranges)!=0):
      line_ranges[0] = (max(0,line_ranges[0][0]-10),line_ranges[0][1])
      line_ranges[-1] = (line_ranges[-1][0],min(image.shape[1],line_ranges[-1][1]+7))

    for i in range(1,len(line_ranges)):
        temp = (line_ranges[i-1][1] + line_ranges[i][0]) // 2
        line_ranges[i-1] = (line_ranges[i-1][0], temp)
        line_ranges[i] = (temp, line_ranges[i][1])

    # Crop and save each detected line
    cropped_lines = []
    for idx, (start, end) in enumerate(line_ranges):
        cropped_line = image[start:end, :]  # Crop the original grayscale image
        cropped_lines.append(cropped_line)
        # Save the cropped line as an image
        # print(cropped_line)
        output_path = f"{output_dir}/line_{idx + 1}.png"
        cv2.imwrite(output_path, cropped_line)
        print(f"Saved cropped line {idx + 1} to {output_path}")