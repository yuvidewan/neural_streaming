"""Video/image-sequence inspection, frame extraction, dataset ingestion,
and the PyTorch data pipeline built on top of it.

- video_utils:      OpenCV VideoCapture wrapper with clear validation errors
- sequence_utils:   folder-of-images (e.g. DAVIS-style) discovery + validation
- frame_extraction: frame sampling, resize/crop, PNG export (shared by both)
- errors:           DatasetSourceError, the shared base for both source types
- sources:          DatasetSource strategy (VideoDatasetSource, ImageSequenceDatasetSource)
- dataset_prep:     original (Milestone 2) video-only pipeline; kept for
                     backward compatibility, still used directly by ingest.py
- ingest:           source-agnostic pipeline (video or image sequence) used
                     by scripts/prepare_dataset.py
- validation:       DatasetValidationError + frame tensor sanity checks
- frame_dataset:    FrameDataset (torch.utils.data.Dataset), reads manifest.json
- transforms:       train/eval transform pipelines for FrameDataset
- loaders:          create_train_loader/create_val_loader/create_test_loader

No neural network, VAE, compression, or streaming code lives here yet.
"""
