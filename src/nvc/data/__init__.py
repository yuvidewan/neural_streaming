"""Video/image-sequence inspection, frame extraction, and dataset ingestion.

- video_utils:      OpenCV VideoCapture wrapper with clear validation errors
- sequence_utils:   folder-of-images (e.g. DAVIS-style) discovery + validation
- frame_extraction: frame sampling, resize/crop, PNG export (shared by both)
- errors:           DatasetSourceError, the shared base for both source types
- sources:          DatasetSource strategy (VideoDatasetSource, ImageSequenceDatasetSource)
- dataset_prep:     original (Milestone 2) video-only pipeline; kept for
                     backward compatibility, still used directly by ingest.py
- ingest:           source-agnostic pipeline (video or image sequence) used
                     by scripts/prepare_dataset.py

A torch.utils.data.Dataset/DataLoader built on top of the extracted PNG
frames is planned for a later milestone and not implemented here.
"""
