# Person-Re-identification
Person re-identification is an image retrieval problem where we wish to retrieve the image of a person from a gallery of images given a query image.

## Introduction
Person re-identification (Re-ID) is a well-known problem in computer vision-based surveillance. Re-ID aims to identify the same Person from a variety of non-overlapping viewpoints from multiple cameras.Re-ID is a challenging task due to the presence of different viewpoints varying low-image resolutions, illumination changes, unconstrained poses, occlusions, heterogeneous modalities, complex camera environments, background clutter, unreliable bounding box generations, etc. These
result in varying variations and uncertainty.

## Locally-Aware Transformer
The primary output of a vision transformer is a global classification token, but vision transformers also yield local tokens which contain additional information about local regions of the image. LA-Transformer combines vision transformers with an ensemble of FC classifiers that take advantage of the 2D spatial locality of the globally enhanced local tokens.
