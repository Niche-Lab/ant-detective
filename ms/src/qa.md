


Thank you for your comments and suggestions. We have revised the manuscript according to your feedback. Below are our responses to your comments:

Effective ant counting systems can be built cheaply with minimal manual effort
Breaking images into smaller patches dramatically improves counting in dense ant colonies
Automated counting reveals long-term foraging patterns that are difficult to capture through manual observation
RetryClaude can make mistakes. Please double-check responses.

###### Because the readers of "Ecological Informatics", including me, are not always familiar with ants, adding a little more explanation about ant foraging behavior and scenarios is recommended.

Authors: Thank you for your comments and we acknowledge the need for more detailed explanations. We have added the requested information in the revised manuscript as a section named "3.2 Experimental Setup and Rationale" (L154-201). This section now provides a more comprehensive overview of the experimental scenarios, including the rationale behind the design choices and the specific conditions under which the experiments were conducted.

###### In addition, discussing a little more about reason(s) why performance for 8 x 10 patches was lower than 4 x 10 patches and possible technical limitations (e.g. effects of occlusion) will attract wider interests of readers.

Authors: Thank you for your comments. We have added a brief explanation on why the specific patch size is better suited than other in L446-450. We also discussed the study limitation by including the external dataset iNaturalist in the section "4.6 Study Limitations and Future Work" (L518-547).

Specific comments
###### L8-13: I recommend adding a little more explanation about scenarios.
Authors: Thank you for your suggestion. We have added brief examples defining the simple and complex scenarios in the abstract (LLLL).

###### L27-30: One example is presented [6], but adding further references about manual counting is recommended for addressing a general problem.

Authors: Thank you for your suggestion, we have added another example of manual observing the ant foraging behavior across multiple locations and species to demonstrate the potential benefits of the ant counting method. (LXXX)

###### L80-83: Add the size of Petri dish. In A02 setting, two Petri dishes with the same size as A02 and A03 were placed in one image?

###### L89-91: Adding a little more explanation about complexity of background is needed. Were debris and parts of insect prey artificially scattered? May the condition usually occur during the experiments for ant behavior? Is the complexity similar in B01 and B02?

Authors: Thank you for the comments. We have added more details of each studied image sets in the section "3.2 Experimental Setup and Rationale" (LLL) to answer your questions.

###### L97-98 and so on: "Alt text" seems to be different from figure legends. For a legend of Fig. 1, more explanations are needed because the meaning of the numbers in Fig. 1a, for example 9.20 (7.37) and 4.59 (3.83) is unclear (average number of ants per image?).

Authors: Thank you for your comments and sorry for the overlook. We have added more details to explain the figure components in Fig. 1.

###### L100-109: "Calibration" dataset was obtained from the experimental setups of A01, A02, and A03 (L79-86) or different setups "similar to" A01, A02, and A03? If the former is true, the naming of experimental setups and datasets is confusing (calibration and A01-A03 data were obtained from setups A01-A03). If the latter is true, an explanation about how the calibration data were obtained is needed.

Authors: Thank you for your comments and sorry for the confusion. The later is true. The calibration dataset was obtained from the experimental setups similar to A01, A02, and A03 but at different time points. We have clarified this in the revised manuscript (LXXX).

###### Table 1: Notes about B01 (tube feeder) and B02 (outdoor) seem not to be consistent with main texts.

Authors: Thank you for your comments. We have revised the notes in Table 1 to be consistent with the main text (LXXX).

###### L127: Original images were resampled, not after augmentation?

Authors: Since the data augmentation that was applied to the images to increase the image diversity, even two images sampled from the same image can be different. We have clarified this in the revised manuscript (LXXX).

###### L140: Why was the resolution different from 1920 x 1080 (L105)?

Authors: Sorry for the confusion. The 'minimum' is redundant here. The original resolution of the images was set to 1920 x 1080, and we have removed the word 'minimum' in the revised manuscript (LXXX).

###### L150: I am a little skeptical about the validity of converting a bounding box to a circle. Is it important to consider the body size of ants as a circle, not just assume ants as points?

Authors: Thank you for your comment. It actually is a parameter of circle radius that can be adjusted based on the body size of ants or activity level of ants. With sparse distribution of ants, the circle radius can be set to a larger value to be more visible in the heatmap. In contrast, with dense distribution of ants, the circle radius can be set to a smaller value or even to a point to avoid overlapping. We have clarified this in the revised manuscript (LXXX).

###### L153-164: Define x and y clearly. In equations 1 and 2, x and y seem to indicate coordinates of pixels, but x and y in equation 3 represent the center of ant detection (x0 and y0 in equation 1).

Authors: Thank you for your comments. We have clarified the definition of x, y, $x_0$, and $y_0$ in the revised manuscript (LXXX).

###### L175: Studies 1 and 2 seem not to be clearly defined.

Authors: Sorry for the confusion. We have revised the section title by replacing "Objective" with "Study" to clarify the definition of Study 1 and Study 2 (LXXX).

###### Fig. 3: Here, the word "training images" is used, but "calibration" is used in the main text.

Authors: Thank you for your comment. We have revised the figure axis title to be consistent with the main text (LXXX).

###### L238-241: The values indicate overall performance? Are there any variations in performance among images (or patches)? Such information will be helpful for understanding the model performance.

Authors: Thank you for your comments. We have revised the figure and the main text by adding the word "average" before performance to avoid misunderstanding. 

###### L250-252: Are there any possible reason(s) why performance for 8 x 10 patches was lower than 4 x 10 patches? Discussing a little more about this phenomenon looks important for generalization and further application of similar methods.

Authors: Thank you for your comments. We have added a brief explanation in the section "4.4 SAHI tuning results on the Dense subset" (LXXX) to discuss the possible reasons. 

###### L263-264: Recommending this possible hypothesis in the M&M section to enhance understanding of experimental setups.

Authors: Thank you for your suggestion. We have added a brief explanation of the hypothesis in the section "3.2 Experimental Setup and Rationale" (LXXX) to enhance understanding of the experimental setups.

###### L274: This study does not indicate that combining spatial and temporal information does not enhance ant detection and counting, but indicates enhancing understanding of ant foraging behavior. Please re-phrase.

Authors: Sorry for the confusion. We have revised the section title accordingly in section 4.5 (LXXX).

###### L282-284 and the following paragraphs: The method will promote effective investigation of ant foraging behavior, but this study itself does not reach robust conclusions about ant behavior because statistical analyses were not performed. Weakening the expression is recommended.

Authors: Thank you for your comments. We have revised the section 4.5 accordingly and added a clarification that a statistical analysis is needed to draw robust conclusions about ant foraging behavior (LXXX).

###### L290-291: Correspondence of left / right dishes and sucrose / peptone should be clarified (indicated by S / P in Fig. 8? But not clarified in the legend).

Authors: Thank you for your comments. We have revised both the figure and the main text to clarify the correspondence of left/right dishes and sucrose/peptone (LXXX).

###### Fig. 8: The heatmap does not illustrate "ant activity over time." Please revise the legend. In addition, probably because the heatmap is overlaid on the original image, lighter places look more white than expected from ant density, potentially leading the misunderstanding. Showing the heatmap not overlaid on the original image may be better.

Authors: Thank you for your comments. We have revised the figure legend to clarify that the heatmap illustrates ant activity over time, and we have also removed the original image in the heatmap to avoid misunderstanding (LXXX).

## --------- Reviewer #3


###### Observation 1: As the paper is present in pre-print version therefore though plagiarism shows self-plagiarism.

Authors: Thank you for spotting this issue. We have put a note to the reviewer that this paper is a pre-print version and the self-plagiarism is not an issue. 

###### Observation 2: The paper will get more insights if the authors could add relevant references of similar kind of work done by image processing for ants or some other social insects in the result and discussion section

Authors: Thank you for your suggestion. We have added more relevant references of similar work, such as animal tracking tool or bird counting from aerial images, in the section "2. Background Study". We have also added a discussion on the limitation by including an external dataset iNaturalist in the section "4.6 Study Limitations and Future Work".

## --------- Reviewer #4

###### 1.Abstract and conclusion need explicitly elaborate the results of the experimental design for foraging behavior, and need specify spatial analysis of ant activity over time, and findings on food preferences and foraging intensity of ants infected with the virus;

Authors: Thank you for your comments. We have revised the manuscript by providing rationale for the experimental design in section "3.2 Experimental Setup and Rationale". We have also revised the abstract and conclusion accordingly.

###### 2.P4, in the experimental setup section, the purpose and significance of each experimental design is not mentioned;

###### 3.P4 L95, the authors chose to take the photographs every day at 5 p.m. Is the choice of time point exceptional and what is the basis for the time choice?

###### 4.P8 L161, the purpose of selecting the two bait types for the B02 subset should be reflected in the Experimental Setup section.;

Authors: Thank you for your comments. For the series of comments (2, 3, 4), we have revised the manuscript by providing rationale for the experimental design in section "3.2 Experimental Setup and Rationale".

###### 5.P11 Table2 , Why are precision and recall rates expressed as mean ±1.96 standard deviation? What is the rationale for using this representation?

Authors: Thank you for your comments. Our original intention was to provide a 95% confidence interval for the mean precision and recall rates. However, we have revised the table to show the mean ± standard deviation instead, as it is more commonly used in the field (LXXX).

## --------- Reviewer #5

###### 1. Comparative Experiments with State-of-the-Art Methods
(1) Issue:
The paper does not compare the proposed method (YOLOv8n + slicing strategy) with advanced techniques for small object detection, such as YOLOv5 with attention mechanisms, the SAHI framework, or transformer-based detection models (e.g., DETR).
(2) Revision Suggestions:
Conduct comparative experiments on the same dataset to evaluate precision (AP), recall (AR), and computational efficiency (FPS) against existing methods, clearly highlighting the advantages of the proposed approach.
Analyze the differences between the slicing strategy and existing frameworks (e.g., SAHI), particularly in handling overlapping slices or post-fusion strategies (e.g., non-maximum suppression, NMS).

Authors’ responses: Thank you for your comments. Our original intention was to focus on promoting this method for ant foraging behavior studies and similar ecological observations. Given our limited expertise in advanced object detection methods, we did not intend to create a new state-of-the-art method. Hence, we have revised the entire manuscript to focus on leveraging the SAHI framework, which is a widely used framework for small object detection, to evaluate the system performance. We also included RT-DETR in the comparison as it is a transformer-based detection model.
We also expanded the evaluation metrics in not only performance (mAP@0.5, F1, Precision, Recall) but also computational efficiency (FPS, GPU memory usage) to assess the real-time applicability of the method.

######  2. Optimization Details of Slicing Strategy and Computational Efficiency Metrics
(1) Issue:
The rationale for selecting specific slice sizes (e.g., 4x10) and the mechanism for handling cross-slice objects (e.g., ants split across patches) are unclear. Additionally, computational costs (e.g., inference speed, GPU memory usage) under different slicing configurations are not quantified.
(2) Revision Suggestions:
Provide a theoretical or empirical basis (e.g., grid search, analysis of ant size relative to patch resolution) for choosing optimal slice dimensions.
Design a cross-slice target fusion algorithm (e.g., enhanced NMS) to mitigate duplicate detections or missed ants at patch boundaries.
Report quantitative metrics, such as inference speed (FPS) and GPU memory consumption, for various slicing configurations (e.g., 2x2, 4x10) to assess real-time applicability.

Authors’ responses: Thank you for your comments. We have revised the manuscript to include both Bayesian optimization method and grid search to select the optimal slice size. We also added a discussion on the advantages of different optimization methods in both performance and computational efficiency.

######  3. Validation of Generalization in Diverse Scenarios
(1) Issue:
The dataset lacks diversity in lighting conditions (e.g., shadows, dynamic illumination) and is limited to a single species (ants), limiting insights into the model's robustness and generalizability.
(2) Revision Suggestions:
Test the model's robustness under varied lighting conditions (e.g., glare, low light) and complex backgrounds (e.g., natural environments with debris).
Validate the method's generalizability on public insect datasets (e.g., iNaturalist) or other species (e.g., bees, beetles) to explore its potential for cross-species or cross-domain adaptation. 

Authors’ responses: Thank you for your comments. We have included an external dataset iNaturalist to validate the method's generalizability. We also added a discussion on the limitations of the study in section "4.6 Study Limitations and Future Work" to address the lack of diversity in lighting conditions and species.


## --------- Reviewer #6

###### Regarding the methodology, the use of YOLOv8 P2 models is appropriate for detecting small objects, such as those in this study. Additionally, techniques like Slicing Aided Hyper Inference (SAHI) exist for enhancing the detection of small objects. It would be beneficial to compare the authors' current methods with such methods, especially since the paper does not reference any studies that use computer vision for ant detection and tracking, which would provide a more comprehensive context for their work.

Authors : Thank you for your comments. As we stated in the response to Reviewer #5, our original intention was to focus on promoting this method for ant foraging behavior studies and similar ecological observations. Given our limited expertise in advanced object detection methods, we did not intend to create a new state-of-the-art method. Hence, we have revised the entire manuscript to directly focus on leveraging the SAHI framework, which is a widely used framework for small object detection, to evaluate the system performance. 

###### Regarding the structure, the authors have combined the Results and Discussion sections into one. While I found this approach did not hinder my understanding of the paper, it may not align with the journal's formatting requirements.

Authors: Thank you for your kind reminder. We have carefully checked the journal's author guidelines and did not find any specific requirements regarding the structure of the Results and Discussion sections. 

I am providing a couple of comments below, hoping that you will consider them and find them useful in improving your manuscript:
###### 1. Line 79: The species' Latin names are mentioned only in the Methods section. It would be beneficial to include them in the Introduction and explain the rationale for selecting these species for the experiments. 

Authors: Thank you for your suggestion. We have provided a brief explanation of the rationale for selecting these species for the experiments in the section "3.2 Experimental Setup and Rationale" (LXXX).

###### 2. Line 79: Could you briefly explain what peptone solutions are and why they were used in the experiments? [This is addressed in the “giant” paragraph above.]

Authors: Thank you for your comment. We have added a brief explanation of peptone solutions and their use in the experiments in the section "3.2 Experimental Setup and Rationale" (LXXX).

###### 3. Lines 116-117: Is there only one species involved in the experiments, or could there be other ant species present? [I think the reviewer may have misunderstood us. Only one ant species was used for every experimental setup (A01-A03&B1-B3), unless I misunderstood the question]

Authors: Thank you for your comment. There are only one ant species involved in the experiments. For more detailed rationale, we have added a brief explanation of peptone solutions and their use in the experiments in the section "3.2 Experimental Setup and Rationale" (LXXX).

###### 4. Lines 89-90: What was the rationale for choosing virus-infected ants (specifically, odorous house ant virus 1, OHAV-1) for comparison with uninfected ones? [This is addressed in the “giant” paragraph above.]

Authors: Thank you for your comment. We have added a brief explanation of the rationale for choosing virus-infected ants (specifically, odorous house ant virus 1, OHAV-1) for comparison with uninfected ones in the section "3.2 Experimental Setup and Rationale" (LXXX).

###### 5. Lines 169-174: It would be beneficial to include a (conceptual) figure illustrating the structure of YOLOv8, along with a brief description of how it operates.

Authors: Thank you for your suggestion. As the model architecture is not the main contribution of this work, and we have not made any modifications to the YOLO model, we have not included a figure illustrating the structure of YOLOv8. We have added references for readers to refer to the original YOLOv8 document for more details on the model architecture and how it operates.

###### 6. Lines 169-174: Did you consider testing the performance of larger models, such as YOLOv8s or YOLOv8m, in addition to YOLOv8n? Given that the reported precision is 77.97% and recall is 71.36% for challenging scenarios, which are not really high in my opinion (as there were no other studies to compare to), larger models could potentially yield better performance. While the current model is suitable for personal computers, labs or companies may deploy your models on servers, where larger models could enhance performance for this intriguing application.

Authors: Thank you for your comment. We have included two larger models, YOLO11m and RTDETR, in the revised manuscript to compare the performance with YOLOv11n. Both models have more than 10 times more parameters than YOLOv11n, and we did see a significant improvement in performance.

###### 7. References: Most of the references are incomplete, missing DOIs or URLs, etc.

Authors: Thank you for your comment. We have provided a complete list of references in BibTex format, which is the standard format for LaTeX documents. We have ensured that every reference is complete and includes DOIs or URLs. Also, we believe that the journal has its own reference format to display the references in the final published version.