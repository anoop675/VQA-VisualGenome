# Visual-Question-Answering-System-VisualGenome

# *Dataset Description:*
Homepage: https://homes.cs.washington.edu/~ranjay/visualgenome/

Dataset Structure: https://huggingface.co/datasets/ranjaykrishna/visual_genome/blob/main/README.md#dataset-structure

Visual Genome is a large-scale dataset designed for vision-and-language research. Its goal is to provide rich annotations connecting images with objects, attributes, relationships, and natural language descriptions.

Size: ~108,000 images

Annotations: Dense and structured, including:
* Objects: Bounding boxes + labels
* Attributes: Object descriptors (e.g., color, shape)
* Relationships: Triplets (subject -> predicate -> object)
* Region descriptions: Freeform sentences about image regions
* Question-Answer (QA) pairs: VQA-style questions
  *  Freeform QAs: Questions about the entire image.
  *  Region-based QAs: Questions about specific regions, collected to study how well models perform at answering questions using the image or the description as input.

Applications:
  Visual Question Answering (VQA),
  Scene graph generation,
  Image captioning,
  Multimodal learning

 # *References:*

[1] Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L. J., Shamma, D. A., Bernstein, M., and Fei-Fei, L. (2017). Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. International Journal of Computer Vision, 123(1), 32-73.

[2] Y. Zheng and J. Lu, "Cross-Modality Encoder Representations Based On External Attention Mechanism," 2023 3rd International Conference on Neural Networks, Information and Communication Engineering (NNICE), Guangzhou, China, 2023, pp. 711-714.
