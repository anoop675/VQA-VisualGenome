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

# *Future Work:*

* Integration of Depth-based Spatial Reasoning

  Current limitation: The existing system primarily leverages 2D image features from ViT and text features from BERT, without explicit modeling of spatial relationships between objects.

  Next steps:

  * Incorporate depth estimation from monocular RGB images using pre-trained depth models (e.g., MiDaS or DenseDepth) to provide per-pixel spatial information.
  * Encode depth-aware features alongside ViT patch embeddings to enable spatial reasoning of object–relation pairs.
  * Explore graph-based positional encodings or 3D bounding boxes to improve relational understanding in complex scenes.

* Enhanced Object–Relation Modeling

  Current limitation: Cross-attention fusion is applied at the sequence level but lacks explicit relational reasoning between detected objects.

  Next steps:
  
  * Implement relational attention mechanisms where attention weights are conditioned on object proximity, co-occurrence, or semantic relationships.
  * Explore integration of scene graphs derived from Visual Genome annotations for guiding attention across objects and relationships.
  * Investigate hybrid approaches combining graph neural networks (GNNs) with cross-attention for robust object–relation reasoning.

* Q-Former–Style Cross-Modal Reasoning

  Current limitation: The current cross-attention fusion is linear and does not support multi-stage query-based reasoning.

  Next steps:
  
  * Introduce a Q-Former–style transformer module that queries visual embeddings based on textual tokens, enabling hierarchical reasoning over visual entities.
  * Experiment with multiple query layers to refine attention over objects relevant to the question context.
  * Benchmark improvements in answer accuracy and reasoning interpretability with Q-Former integration.

* Scalable Multi-Modal Pipeline Enhancements

  Current limitation: Embeddings are precomputed and stored locally, limiting scalability for larger datasets.

  Next steps:
  
  * Design an end-to-end pipeline with on-the-fly embedding extraction, caching, and batching to handle the full Visual Genome dataset (~108k images).
  * Integrate distributed training strategies using PyTorch Lightning or Hugging Face Accelerate to accelerate model convergence.
  * Implement mixed-precision training and gradient checkpointing for memory efficiency.

* Robust Evaluation Metrics and Real-world Benchmarking

  Current limitation: Evaluation is based primarily on accuracy, precision, recall, and F1 on limited top-K answers.

  Next steps:
  
  * Extend metrics to cover relational reasoning accuracy, object–relation alignment, and answer grounding.
  * Perform ablation studies to quantify the contribution of depth, relational attention, and Q-Former modules.
  * Validate the system on downstream tasks such as multi-object question answering and scene graph–based reasoning to assess enterprise applicability.

* Modular and Extensible Codebase

  Current limitation: The current code is modularized for embeddings and cross-attention fusion, but new modules for depth reasoning and relational attention are not yet integrated.

  Next steps:
  
  * Extend the existing modular framework to include depth processing, scene graph generation, and relational attention modules.
  * Provide clear interfaces for swapping backbone models (ViT, Swin, or CLIP) and text encoders (BERT, RoBERTa).
  * Ensure reproducibility, logging, and checkpointing are standardized for production-ready deployment.

 # *References:*

[1] Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L. J., Shamma, D. A., Bernstein, M., and Fei-Fei, L. (2017). Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. International Journal of Computer Vision, 123(1), 32-73.

[2] Y. Zheng and J. Lu, "Cross-Modality Encoder Representations Based On External Attention Mechanism," 2023 3rd International Conference on Neural Networks, Information and Communication Engineering (NNICE), Guangzhou, China, 2023, pp. 711-714.

[3] Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. Proceedings of the 40th International Conference on Machine Learning, 202, 19730–19742.

[4] Adam Santoro, David Raposo, David G. T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter W. Battaglia, Tim Lillicrap. A simple neural network module for relational reasoning. Advances in Neural Information Processing Systems (NIPS), pp. 4967-4976, 2017.

[5] P. Anderson et al., “Bottom-up and top-down attention for image captioning and visual question answering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2018, pp. 6077–6086.

[6] Liu, Y., Wei, W., Peng, D., Mao, X., He, Z. Depth-Aware and Semantic Guided Relational Attention Network for Visual Question Answering. IEEE Transactions on Multimedia, 2022, pp. 1-14.

[7] R. Cadène, H. Ben-younes, M. Cord, and N. Thome, “MUREL: Multimodal relational reasoning for visual question answering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 1989–1998.

[8] A. Fukui et al., “Multimodal compact bilinear pooling for visual question answering and visual grounding,” in Proc. Conf. Empir. Methods Natural Lang. Process., 2016, pp. 457–468.

[9] Kamath, A., Singh, M., LeCun, Y., Synnaeve, G., Misra, I., & Carion, N. MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding. arXiv preprint arXiv:2104.12763, 2021.
