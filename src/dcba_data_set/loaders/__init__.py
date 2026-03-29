"""Data loading primitives for DCBA datasets.

The :class:`~dcba_data_set.loaders.data_models.DCBAHeteroData` and
:class:`~dcba_data_set.loaders.data_models.ConfigRecord` classes form the data model.
:func:`~dcba_data_set.loaders.abcd_loader.load_abcd_report` parses a dataset produced
by :class:`~dcba_data_set.ds_generator.DatasetGenerator` into these objects.

The ``torch.utils.data.Dataset`` wrapper that zips graphs and configs into
``(DCBAHeteroData, ConfigRecord)`` pairs for training lives in the training repository.
"""

from dcba_data_set.loaders.abcd_loader import load_abcd_report
from dcba_data_set.loaders.data_models import ConfigRecord, DCBAHeteroData

__all__ = ["ConfigRecord", "DCBAHeteroData", "load_abcd_report"]
