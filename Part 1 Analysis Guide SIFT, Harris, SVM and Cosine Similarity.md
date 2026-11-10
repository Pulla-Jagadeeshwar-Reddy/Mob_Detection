# Part 1 Analysis Guide: SIFT, Harris, SVM and Cosine Similarity

Minecraft mob classification (cow, creeper, pig, zombie). This guide explains what `part1_ml.py` does and how to interpret its numbers, so you can explain the analysis without reading the code.

## 1. The big picture

An SVM needs every image to be described by a fixed-length list of numbers. A photo has no such list, and different photos have different numbers of interesting points. The solution is the **Bag of Visual Words (BoVW)** method:

1. Find interesting points in each image (keypoints).
2. Describe each point with 128 numbers.
3. Group all descriptors into a small number (k) of typical patterns, the "visual words".
4. Describe each image by how often each visual word appears. This gives every image one list of k numbers.
5. Train an SVM on those lists.

The analysis compares two ways of doing step 1: **SIFT** and **Harris corners**.

## 2. Step 1: finding keypoints

**Harris corner detector.** It looks for places where the image changes strongly in *both* directions, which are corners. A flat area or a straight edge does not qualify. It outputs only locations, nothing about what the area looks like. It is not designed to handle size changes: a corner in a small mob and the same corner in a large mob look different to it.

**SIFT detector.** It looks for blob-like points across many zoom levels (scales), so the same feature is found whether the mob is near or far. Each keypoint also gets a size and a direction.

**What to expect on Minecraft images.** Textures are blocky and low-resolution, so there are fewer keypoints than in a photograph, and many corners are block edges rather than meaningful features. Report the average number of keypoints per image (the script prints it) and the number of images with none.

## 3. Step 2: describing keypoints

Harris only finds locations, so it cannot describe them. We therefore describe **both** detectors' keypoints with the **SIFT descriptor**: a 128-number summary of the pattern of edge directions in a small patch around the point. It stays similar under small changes of brightness and rotation. Because the descriptor is identical, any difference in results comes from the *detector* alone. This is the key point to state in the report.

## 4. Step 3: building the vocabulary

All descriptors from the **training images** are pooled and grouped with **k-means** into k clusters. Each cluster centre is a visual word, for example "a dark-to-green edge" or "a square corner". The script tries k = 25, 50 and 100.

Two things to remember:

- The vocabulary is built using only training images in each fold. Using test images would leak information and inflate accuracy.
- Small k merges different patterns together. Large k splits patterns so finely that histograms become sparse and noisy, especially with only 100 images.

## 5. Step 4: image histograms

Every descriptor in an image is assigned to its nearest visual word, and the counts form a histogram of length k. The histogram is then scaled to length 1 (L2 normalisation), so images with many keypoints and images with few can be compared fairly.

## 6. Step 5: SVM and accuracy

A **Support Vector Machine** finds the boundary that separates the classes with the widest margin. We try a linear boundary and an RBF (curved) boundary, and a grid search chooses the settings C (how strictly to avoid mistakes) and gamma (how curved the boundary is).

**Why 5-fold cross-validation.** With 25 images per class, a single train/test split would leave only about 5 images per class for testing, so the accuracy would swing wildly depending on which images were chosen. Instead:

- The 100 images are split into 5 groups of 20 (5 per class).
- The model is trained on 4 groups (80 images) and tested on the remaining one.
- This is repeated 5 times so every image is tested exactly once.
- The reported **accuracy** is the average over the 5 folds, with the standard deviation. Accuracy = correct predictions / total predictions.

**Nested search.** The SVM settings are chosen by a second, inner cross-validation using only the training part of each fold. The test fold is never used for choosing settings.

**Confusion matrix.** A 4x4 table: rows are the true class, columns the predicted class. The diagonal is correct predictions. Off-diagonal numbers show which mobs get confused, for example a cow predicted as a pig. Look for mistakes that make visual sense (cow and pig are both four-legged and similarly coloured).

## 7. Step 6: cosine similarity

Cosine similarity measures how alike two histograms are by the angle between them:

cos(a, b) = (a . b) / (|a| x |b|)

It is 1 when two images have the same pattern of visual words, and close to 0 when they share almost none. Because it compares *patterns* rather than totals, it does not matter how many keypoints each image had.

The script uses it in three ways:

1. **Class-versus-class similarity heat-map** (`part1_similarity.png`). For each test image, the average cosine similarity to the training images of each class is computed, then averaged over all test images of the same true class. The diagonal is same-class similarity, the off-diagonal cells are different-class similarity.
2. **Same-class vs different-class summary.** The mean of the diagonal against the mean of the off-diagonal cells. A good feature set has a large gap. A small gap means the features do not separate the mobs well.
3. **Cosine 1-nearest-neighbour accuracy.** Each test image is given the class of the single most similar training image. This is a simple classifier with no training, and it shows whether the features themselves are good, independent of the SVM.

## 8. How to read and write up the results

Questions your report should answer, using `results/part1_summary.csv` and the figures:

- **Which detector gave higher accuracy, SIFT or Harris?** Use the best k for each. Mention the standard deviation: if the gap is smaller than the spread across folds, say the difference is not clear.
- **Why?** SIFT is scale-aware and finds more distinctive points. Harris finds corners only, which on blocky Minecraft textures often sit on block edges.
- **What effect did k have?** Compare the k rows for each detector.
- **Which mobs were confused?** Use the confusion matrix and the similarity heat-map. Cells with high off-diagonal similarity match the confusion matrix errors.
- **Does the SVM agree with cosine 1-NN?** If both are high, the features are good. If 1-NN is much lower, the SVM is compensating with its flexible boundary.
- **Limits.** Only 100 images, one game and one art style, a hand-chosen number of Harris corners, and BoVW ignores where in the image each pattern appears.

## 9. Questions you might be asked

**Why not feed raw pixels to the SVM?** Pixels change completely if the mob moves, scales or the lighting changes. Keypoint descriptors are designed to be stable under those changes.

**Why are the Harris corners described with SIFT?** Harris has no descriptor. Using the same descriptor makes the detector the only difference between the two experiments.

**Why k-means?** It finds the most common descriptor patterns automatically, with no labels.

**What is the risk of fitting the vocabulary on all images?** Test images would influence the features, giving an optimistic accuracy (data leakage).

**What does an accuracy of 0.80 mean here?** On average, 80% of unseen images were classified correctly. Random guessing among 4 classes would give 25%.

**Why is cosine similarity used instead of Euclidean distance?** It ignores the overall length of the histogram and focuses on the pattern, which suits normalised word counts.

## 10. Glossary

- **Keypoint:** a location in the image chosen by a detector.
- **Descriptor:** the list of numbers describing the neighbourhood of a keypoint (128 for SIFT).
- **Visual word:** the centre of a descriptor cluster.
- **Vocabulary:** the set of k visual words.
- **Histogram:** counts of how often each visual word appears in one image.
- **SVM:** a classifier that finds the widest-margin boundary between classes.
- **Cross-validation:** repeated train/test splits so every image is tested once.
- **Data leakage:** accidentally letting test information influence training.
