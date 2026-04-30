# Privacy-Preserving Image Geolocation (LVLM)
Final-year machine learning dissertation (First Class Honours).

A PyTorch-based multimodal Large Vision Language Model (LVLM) that predicts UK image locations from street view images, while evaluating the impact of privacy-preserving techniques such as image and text blurring.

This project demonstrates real-world multimodal AI system design with a focus on responsible machine learning development.

### Tech Stack
PyTorch, Transformers, ViT, BERT, BLIP-2, EasyOCR, OpenCV, CUDA

### Features:
- ViT for image embedding
- BERT, BLIP-2 captions, and EasyOCR for text embedding
- Cross-attention fusion of image and text embedding
- Geocell classification and coordinate regression using a custom multi-layer perceptron (MLP)
- Hybrid loss function of cross-entropy and Haversine distance
- Privacy-preserving experiments using blurring techniques.

### Dataset and Training:
- 40,000 UK street-view images processed and augmented
- GPU accelerated training (CUDA)
- Batch processing with an optimising data pipeline
- Baseline vs privacy-preserving model comparison

### Report
- Full end-to-end system design and implementation
- Experimental evaluation and validation
- Analysis of accuracy vs privacy trade-offs

### Skills Demonstrated
- End-to-end large vision language model (LVLM) design and implementation
- Multimodal learning with image-text fusion
- Transformer architectures (ViT, BERT, BLIP-2)
- Computer Vision preprocessing (OCR, OpenCV, image blurring)
- PyTorch model training, optimisation, and evaluation
- Geospatial machine learning using geocell classification and coordinate regression
- GPU-accelerated training and large-scale dataset handling
- Privacy-aware and responsible AI system design and experimentation

### Notes:
- Source code only (Dissertation.py)
- Trained models and datasets not included

