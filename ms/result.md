# Results and Discussion

## Relationship between Model Performance and Calibration Sample Sizes

The model performance in terms of precision and recall is presented in Table 2 and Figure 3. With a calibration set consisting of only 64 images, similar in background to the test images, the model demonstrates reasonably good performance on subsets A01, A02, and A03. The average precision achieved is 87.96%, 76.01%, and 76.76%, respectively, while the average recall is 87.78%, 77.58%, and 70.23%, respectively. However, increasing the calibration set size does not consistently improve model performance. For instance, in subset A03, doubling the calibration size from 128 to 256 images results in only a slight improvement of 0.22% in precision and 0.41% in recall. Even when increasing the calibration size twentyfold, from 64 to 1024 images, the maximum gains observed are 7.54% in precision for subset A02 and 11.64% in recall for subset A03. In subset A01, these improvements are even smaller, at just 5.14% and 5.33%, respectively. When detection an image containing 20 ants, the 5% improvement in precision and recall corresponds to correctly detecting just one additional ant. These findings suggest the presence of a saturation point in model performance; when the detection task is relatively straightforward and the background is consistent, a small number of calibration images is sufficient to achieve good performance. 

In contrast, for more complex and diverse backgrounds, the relationship between sample size and model performance varies significantly between subsets B01 and B02. In subset B01, the increase in performance is almost linear as the calibration size doubles, with precision and recall improving by 14.64% and 13.09%, respectively. Notably, when all calibration images are utilized, the model’s performance on B01, despite the new and complex background, is nearly equivalent to that on subsets A02 and A03, which feature similar backgrounds. This suggests that deploying the CV system in a new environment requires a large calibration set to ensure the model can generalize effectively to novel images.

Interestingly, in subset B02, characterized by sparse ant distribution and uneven background colors, increasing the calibration size does not significantly enhance model performance. In fact, when the calibration size exceeds 512 images, precision actually decreases. This finding indicates that the model may overfit to the calibration set when the background is highly diverse, leading to generating more false detections and reducing precision.

When comparing manual and automated counting results (Figure 4) using a calibration size of 1024, all subsets except B02 show a high correlation between the automated and manual counts, with a minimum squared correlation (r²) of 0.94. For subsets with similar backgrounds, the RMSE values are 0.96, 1.55, and 1.23 for A01, A02, and A03, respectively. In subset B01, where the background includes multiple black stains resembling ants, the automated counts achieve an r² of 0.95 but with a higher RMSE of 11.36, indicating an overestimation by approximately 11 ants on average. This overestimation is primarily due to false detections caused by background stains that resemble ants.

The poorest performance is observed in subset B02, where the RMSE is 1.50 and r² drops to 0.58. This suggests that the sparse and small distribution of ants makes the counting results highly sensitive to false detections, significantly affecting the correlation. These findings highlight the importance of background uniformity and ant distribution in ensuring accurate automated counting results.


## Slicing Dense-populated Images Significantly Improves Model Performance

The computer vision (CV) system achieved a substantial accuracy improvement by slicing densely populated images into smaller patches (Figure 5, Table 3). Peak performance was observed when the original images were divided into 4 x 10 patches, resulting in a precision of 77.97% and a recall of 71.36%. This represents a significant enhancement compared to the performance on the original images (precision: 9.92%, recall: 1.60%). Additionally, since the YOLOv8n model architecture requires the input image to be in a 640x640 matrix, the slicing ratio plays a crucial role in preventing image distortion, especially when the width/height ratio of the input image deviates from 1. For instance, in this study, the original image size was 1636x2180 pixels. Directly resizing this image to 640x640 results in a height-axis distortion of 1.34 times (calculated from the width/height ratio of 1636/2180). With this in mind, the study also examined whether such height-axis distortion, ideally close to 1, would affect detection performance. Surprisingly, the 4 x 10 patched images, which exhibited the most distorted ratio of 0.53, achieved the best performance, while the 8 x 10 patched images, which were the least distorted at 1.06 times, performed the worst among the sliced images. This indicates that height-axis distortion has no significant correlation with detection performance. Figure 6 provides an example of detection results between the original image and the 4 x 10 patches. Most missed detections in the original image occurred where ants were densely populated, such as on the edges of the nest and the water feeder. This example demonstrates the effectiveness of the slicing strategy, as only a few ants were missed in the 4 x 10 patched images.

The automated counting results for the 4 x 10 patched images were compared with manual counts, showing a high correlation with an R² of 0.938 and an RMSE of 465.05 (Figure 7a, left). The high RMSE value is attributed to overlooked ants in the manual counts, while the CV system was able to capture these missing ants, maintaining strong agreement with the manual counts. The background of experiment B03 was to investigate the impact of SINV-3 infection on ant foraging behaviors. The automated counting results effectively captured the ant population dynamics in the presence of SINV-3 infection (Figure 7a, right). Moreover, when propagating the automated counts to the entire image set collected over the 14-day experiment, a temporal trend of ant population dynamics was observed, which closely matched the manual counts (Figure 7b). Both counting methods revealed a similar trend in ant population dynamics, with SINV-3 infected ants consistently showing higher and more active foraging behavior than the control group. This result demonstrates the potential of the CV system to accurately capture ant population dynamics in a high-throughput manner, which is crucial for studying the effects of pathogens on ant foraging behaviors.



1636x2180: 1.34
818x1090: 1.34
818x545: 0.67
409x545: 1.34
409x218

168/204.5

to ensure different size of patch is not stretched to differen
1636x2180

## Data dissemination and web-based application

The studied ant images with YOLO annotation is organized at 