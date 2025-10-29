### Blur Detection Threshold Adjustment

**Problem:**
The system was failing to discard certain blurry images, specifically those with large, low-detail areas and potential compression artifacts, despite visual blurriness. Two specific instances were identified: `detected_cam1921683111_2025-10-28_14-49-49.jpg` and `detected_cam1921683111_2025-10-28_16-59-47.jpg`.

**Cause:**
The Laplacian variance threshold for blur detection in `rail_obstacle.py` was set to `100`. While Laplacian variance is a suitable metric for blur, this threshold was found to be too low for the specific characteristics of the problematic images. Analysis showed these blurry images had Laplacian variance values of `177.79` and `137.63`, which were above the original threshold, causing them to be incorrectly processed.

**Solution:**
The Laplacian variance threshold in `rail_obstacle.py` was initially increased from `100` to `200`. Subsequently, it was further increased to `400` to address cases where visually blurry images, such as `detected_cam1921683120_2025-10-29_09-54-29.jpg` (Laplacian Variance: 378.80), were not being discarded by the previous threshold. This change makes the blur detection even more strict.

**Verification:**
**Verification:**
A test script was used to analyze the Laplacian variance of problematic blurry images and known sharp images.
*   Initially, with the threshold of `200`, blurry images with variances `177.79` and `137.63` were correctly identified as blurry and discarded.
*   However, an image `detected_cam1921683120_2025-10-29_09-54-29.jpg` with a Laplacian variance of `378.80` was visually blurry but not discarded by the `200` threshold.
*   With the updated threshold of `400`, this image (variance `378.80`) will now be correctly identified as blurry and discarded.
*   A known sharp image (`detected_cam1921683120_2025-10-28_16-30-55.jpg`, with a variance of `1160.63`) remains well above the new threshold, ensuring it is not incorrectly discarded.

**File Modified:**
`rail_obstacle.py`
