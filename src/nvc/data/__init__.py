"""Video inspection, frame extraction, and dataset preparation utilities.

- video_utils:    OpenCV VideoCapture wrapper with clear validation errors
- frame_extraction: per-video frame sampling, resize/crop, PNG export
- dataset_prep:   video discovery, train/val/test splitting, manifest generation

A torch.utils.data.Dataset/DataLoader built on top of the extracted PNG
frames is planned for a later milestone and not implemented here.
"""
