# ConvNeXt for Alzheimer's Screening
## What is ConvNeXt?
ConvNeXt is a modern Convolutional Neural Network (CNN) architecture introduced to bridge the performance and design gap between traditional CNNs and the newer transformers. While transformers dominate computer vision benchmarks due to advanced design choices, they often lack the efficiency, simplicity, and inductive biases of convolutional networks. ConvNeXt bridges this gap by proving that traditional ConvNets can be improved to match or exceed transformer performance while remaining simple and efficient.

## How Does it Work?
ConvNeXt is a highly optimised CNN that processes image data through a series of convolutional stages. It starts off with a stem that groups pixels into patches, passing them through feature blocks where inverted bottlenecks expand and contract channel dimensions to maintain representational power. To capture broader spatial context, ConvNeXt uses large 7x7 depthwise convolutions to isolate spatial filtering from channel mixing. It goes through multiple stages of blocks between downsampling, and integrates design choices such as replacing traditional batch normalisation with layer normalisation and using the GELU activation function, and removes excessive activation layers. This streamlined pipeline enables ConvNeXt to efficiently extract features and maintain a high spatial awaerness through standard convolution operations. See [the original paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) for more details.

<img width="2089" height="598" alt="image" src="https://github.com/user-attachments/assets/037cf03d-c7c0-466c-b0a1-80009d7c2bda" />

The architecture of ConvNeXt (figure by [Lu et al.](https://xiahepublishing.com/m/2310-8819/JCTH-2025-00631))


## Feasibility Review
### 1. User Need, Scope & Acceptance Criteria
This prototype is for medical practitioners/researchers to reliably classify MRIs to determine whether the patients have Alzheimer's, and to provide the level of uncertainty on ambiguous boundary cases.
The acceptance criteria are as follows:
- No data leakage – all MRI slices from one patient must be in the same split
- Peak GPU VRAM of no more than 8GB
- Median inference latency of no more than 100ms per MRI slice
- Per class precision, recall, macro-F1, and AUROC – ConvNeXt should improve on these when compared to the baseline CNN

### 2. Model Choice & Course Concepts
The dataset will be split into 70% training, 15% validation, and 15% testing, ensuring that there is no data leakage between the sets. Data leakage is important to avoid, as Fung et al. (https://arxiv.org/html/1906.04231v1) identified substantial differences in performance with and without patient-level splitting. 
The baseline model will be a small CNN with the following architecture: 
- Input: Single-channel (grayscale) 2D image tensor.
-  Block 1: Conv2d (1 -> 16 channels, 3x3 kernel, padding 1), batch normalisation and ReLU activation, with 2x2 max pooling. Halves spatial dimensions.
-  Block 2: Conv2d (16 -> 32 channels, 3x3 kernel, padding 1), batch normalisation and ReLU activation, with 2x2 max pooling. Halves spatial dimensions again.
-  Block 3: Conv2d (32 -> 64 channels, 3x3 kernel, padding 1), batch normalisation and ReLU activation, with 2x2 max pooling and then adaptive average pooling. Forces spatial dimensions to 1x1 globally.
-  Classifier: convert the resulting 64x1x1 tensor into a 64-element feature vector and output the classification logits.

ReLU activation was chosen because it introduces non-linearity, and is simple and efficient (lecture). Using batch normalisation because it enables faster and more stable training (http://arxiv.org/abs/1805.11604). 

The ConvNeXt prototype will have the following architecture (based on the architecture in “A ConvNet for the 2020s” https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) :
- Input: Single-channel (grayscale) 2D image tensor.
-  Stem: Conv2d (1 -> 32 channels, 4x4 kernel, stride 4) and group normalisation. Halves spatial dimensions.
-  Stage 1 (32 channels): One ConvNeXtBlock module (using depth-wise convolution with a 7x7 kernel, layer normalisation, 1x1 channel expansion by 2, GELU activation, and 1x1 projection, with a residual connection).
-  Downsample 1: Conv2d (32 -> 64 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 2 (64 channels): One ConvNeXtBlock module.
-  Downsample 2: Conv2d (64 -> 128 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 3 (128 channels): One ConvNeXtBlock module.
-  Downsample 3: Conv2d (128 -> 256 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 4 (256 channels): One ConvNeXtBlock module, followed by adaptive average pooling forcing spatial dimensions to 1x1 globally.
-  Classifier: linear layer to output the classification logits.

Using AdamW optimiser for both models because of the consistent regularisation and better generalisation (lectures).

The chosen difficulty band is hard. 

### 3. Preliminary Feasibility Evidence
Initial data audit: 
- Number of participants: 1526
- Scans per class show that the classes are pretty well balanced:
AD in the training/validation class: 10400
NC in the training/validation class: 11120
AD in the test class: 4460
NC in the test class: 4540
- Verified that there is no data leakage between the training/validation set and the test set
- Image dimensions: 256x240
- Missing or invalid data: None
- The mean and standard deviation of the training set is 0.1156 and 0.2229, respectively. This will be used to normalise the data. 

### 4. Risks, Budget & Fallback
Technical risks: limited generalisation if it learns dataset-specific patterns, if data leakage then inflated test accuracy from same patient appearing in multiple splits.

Computing budget:
- Maximum 8GB peak VRAM
- Batch size 32
- Training time: up to 2 hours
- Total initial computation: up to 4 hours

Next planned experiment: establish whether ConvNeXt has a measurable improvement on baseline CNN

Workable fallback: simplify training by reducing batch size to 16 or 8, if calibration is poor, refer more cases for specialist review



• Any dependencies required, including exact versions, and address reproducibility of results.



• Example inputs, outputs, and visual plots of your algorithm.



• Description of any specific pre-processing used, with references if any. Justify your training, validation, and testing splits of the data.



• Investigation of the Open Research Dilemma:
– Quantitative benchmarking of your chosen model against your implemented baseline model under identical evaluation splits.
– Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter count, and inference runtime/latency).
– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset (diagnosing root triggers and failure modes).
– Actionable engineering recommendation and trade-off synthesis for your project manager.

• The mandatory Artificial Intelligence Usage Disclosure subsection - documenting:
• Which AI tools were utilized (e.g., GitHub Copilot for auto-complete, ChatGPT/Claude for debugging syntax, Gemini for concept lookup).
• What specific tasks AI assisted with (e.g., scaffolding data loading functions, refactoring plotting scripts).
• How you audited, verified, and empirically tested the AI-suggested code
ChatGPT was used for helping scaffold the data loading functions. I verified the accuracy by verifying that there was no data leakage (see screenshot below).
![alt text](recognition/ConvNeXt-s4801892/images/verification-of-data.png)


Marking Criteria:
1. Description and explanation of the working principles of the algorithm implemented and the problem it solves (5 Marks).
2. Description of usage and comments throughout scripts (3 Marks).
3. Proper formatting using GitHub markdown (2 Marks).
