# Human-Protein-Atlas-Single-Cell-Classification
Find individual human cell differences in microscope images


There are billions of humans on this earth, and each of us is made up of trillions of cells. Just like every individual is unique, even genetically identical twins, scientists observe differences between the genetically identical cells in our bodies.

Differences in the location of proteins can give rise to such cellular heterogeneity. Proteins play essential roles in virtually all cellular processes. Often, many different proteins come together at a specific location to perform a task, and the exact outcome of this task depends on which proteins are present. As you can imagine, different subcellular distributions of one protein can give rise to great functional heterogeneity between cells. Finding such differences, and figuring out how and why they occur, is important for understanding how cells function, how diseases develop, and ultimately how to develop better treatments for those diseases.

To see more, start with less. That may seem counterintuitive, but the study of a single cell enables the discovery of mechanisms too difficult to see with multi-cell research. The importance of studying single cells is reflected in the ongoing revolution in biology centered around technologies for single cell analysis. Microscopy offers an opportunity to study differences in protein localizations within a population of cells. Current machine learning models for classifying protein localization patterns in microscope images gives a summary of the entire population of cells. However, the single-cell revolution in biology demands models that can precisely classify patterns in each individual cell in the image.

The Human Protein Atlas is an initiative based in Sweden that is aimed at mapping proteins in all human cells, tissues, and organs. The data in the Human Protein Atlas database is freely accessible to scientists all around the world that allows them to explore the cellular makeup of the human body. Solving the single-cell image classification challenge will help us characterize single-cell heterogeneity in our large collection of images by generating more accurate annotations of the subcellular localizations for thousands of human proteins in individual cells. Thanks to you, we will be able to more accurately model the spatial organization of the human cell and provide new open-access cellular data to the scientific community, which may accelerate our growing understanding of how human cells functions and how diseases develop.

This is a weakly supervised multi-label classification problem and a code competition. Given images of cells from our microscopes and labels of protein location assigned together for all cells in the image, Kagglers will develop models capable of segmenting and classifying each individual cell with precise labels. If successful, you'll contribute to the revolution of single-cell biology!

The scientific journal Nature Methods is interested in considering a paper discussing the outcome and approaches of the challenge. The Human Protein Atlas team, led by Professor Emma Lundberg, would like to invite top performing teams to join as co-authors in writing this paper. Please follow the discussion forum for more details on how you can help.


Submission File

For each image in the test set, you must predict a list of instance segmentation masks and their associated detection score (Confidence). The submission csv file uses the following format:

ImageID,ImageWidth,ImageHeight,PredictionString
ImageAID,ImageAWidth,ImageAHeight,LabelA1 ConfidenceA1 EncodedMaskA1 LabelA2 ConfidenceA2 EncodedMaskA2 ...
ImageBID,ImageBWidth,ImageBHeight,LabelB1 ConfidenceB1 EncodedMaskB1 LabelB2 ConfidenceB2 EncodedMaskB2 …

Note that a mask MAY have more than one class. If that is the case, predict separate detections for each class using the same mask.

ImageID,ImageWidth,ImageHeight,PredictionString
ImageAID,ImageAWidth,ImageAHeight,LabelA1 ConfidenceA1 EncodedMaskA1 LabelA2 ConfidenceA2 EncodedMaskA1 ...

A sample with real values would be:

ID,ImageWidth,ImageHeight,PredictionString
721568e01a744247,1118,1600,0 0.637833 eNqLi8xJM7BOTjS08DT2NfI38DfyM/Q3NMAJgJJ+RkBs7JecF5tnAADw+Q9I
7b018c5e3a20daba,1600,1066,16 0.85117 eNqLiYrLN7DNCjDMMIj0N/Iz9DcwBEIDfyN/QyA2AAsBRfxMPcKTA1MMADVADIo=

The binary segmentation masks are run-length encoded (RLE), zlib compressed, and base64 encoded to be used in text format as EncodedMask. Specifically, we use the Coco masks RLE encoding/decoding (see the encode method of COCO’s mask API), the zlib compression/decompression (RFC1950), and vanilla base64 encoding.

An example python function to encode an instance segmentation mask would be:

import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib


def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str


Submissions to this competition must be made through Notebooks. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

    CPU Notebook <= 9 hours run-time
    GPU Notebook <= 9 hours run-time
    TPUs will not be available for making submissions to this competition. You are still welcome to use them for training models. For a walk-through on how to train on TPUs and run inference/submit on GPUs, see our TPU Docs.
    No internet access enabled on submission
    External data, freely & publicly available, is allowed. This includes pre-trained models.
    Submission file must be named submission.csv
