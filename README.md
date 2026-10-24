# hackathon_final_project
final project Waste classification
Waste Classification with Deep Learning (PyTorch)

An image classifier that sorts waste into 12 categories using a CNN trained from scratch, compared against a transfer-learning baseline (ResNet18). Built and trained in Google Colab on a T4 GPU.

Dataset
Source: Garbage Classification (Kaggle, mostafaabla/garbage-classification), downloaded with kagglehub
Size: 15,515 images
Classes (12): battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
Imbalance: clothes is the largest class at 34.3% of all images
Split: 70% train / 15% validation / 15% test (10,860 / 2,327 / 2,328), random with a fixed seed (42)
-------------------------------------------------------------------------------------------------
I used a CNN architecture → trained from scratch in PyTorch on 128×128 RGB images, with four convolutional blocks:

Conv2d (3×3, padding=1) to extract features (edges, then textures, then shapes) → BatchNorm2d to stabilize and speed up training → ReLU for non-linearity → MaxPool2d (2×2) to halve the spatial size and keep the strongest features. The channels grow 32 → 64 → 128 → 256 across the blocks.

→ AdaptiveAvgPool2d (4×4) to shrink the final feature map to a fixed size → Flatten (256×4×4 = 4096 values) → Linear (4096 → 256) → ReLU → Dropout (0.4) to reduce overfitting → Linear (256 → 12) to output one logit per waste class.

→ CrossEntropyLoss (softmax + negative log-likelihood) to measure the error → loss.backward() (backpropagation) to compute gradients → Adam optimizer (lr = 0.001) to update the weights, over 15 epochs with batch size 128.

→ Validation after every epoch, saving the best checkpoint by validation accuracy, then a final test on unseen data: 81.40% accuracy, 0.753 macro F1.

→ Compared against transfer learning with a pretrained ResNet18 (new 12-class final layer), which reached 69.20% test accuracy. The CNN looks better here mainly because of the too-high learning rate in the ResNet run, so don’t present it as proof that training from scratch wins.
-----------------------------------------------------------------------------------------
Business Idea Summary

Problem: People sort waste incorrectly, so recycling fails, and manual sorting is slow, costly, and unsafe for workers.

Solution: A smart waste-sorting assistant. A camera photographs the item, and our own CNN (81% accuracy) identifies it as one of 12 waste types and tells the user which bin it belongs in. The model is lightweight (about 1.6M parameters), so it runs offline on cheap hardware like a Raspberry Pi.

Customers:

Schools, universities, and malls (smart bins)
Municipalities and recycling companies (less manual labor)
Companies that need waste reports

Revenue: A monthly subscription for the software and dashboard, hardware sales or leasing for the smart bins, and per-call pricing for API access.

Roadmap: Start with a mobile app as the MVP, then smart bins, then industrial sorting lines. We’re also improving the weaker classes like metal and plastic, and prioritizing batteries because a misclassification there is a safety risk.
