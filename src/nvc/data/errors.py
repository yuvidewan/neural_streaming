"""Shared error base for all dataset source implementations.

Both the video pipeline (video_utils.VideoError) and the image-sequence
pipeline (sequence_utils.SequenceError) inherit from this, so orchestration
code (ingest.py) can catch validation/read failures generically regardless
of which kind of source produced them.
"""


class DatasetSourceError(Exception):
    """Base class for errors raised while validating/reading one dataset item."""
