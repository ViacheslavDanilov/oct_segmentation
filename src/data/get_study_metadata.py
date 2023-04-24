import logging
import os
from datetime import datetime

import hydra
import pandas as pd
import pydicom
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_file_list, get_series_name, get_study_name

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def extract_metadata(
    dcm_path: str,
) -> dict:

    dcm = pydicom.dcmread(dcm_path)
    keys = [
        'Path',
        'Study UID',
        'Series UID',
        'Study name',
        'Series name',
        'Acquisition Date',
        'Acquisition Time',
        'Patient Name',
        'Patient Sex',
        'Body Part',
        'Physician',
        'Institution',
        'Manufacturer',
        'Modality',
        'Image Type',
        'Height',
        'Width',
        'Slices',
        'Channels',
        'Data Type',
        'WC',
        'WW',
    ]
    meta = {key: float('nan') for key in keys}
    meta['Path'] = dcm_path
    meta['Study UID'] = str(dcm.StudyInstanceUID)
    meta['Series UID'] = str(dcm.SeriesInstanceUID)
    meta['Study name'] = get_study_name(dcm_path)
    meta['Series name'] = get_series_name(dcm_path)

    try:
        if hasattr(dcm, 'AcquisitionDate'):
            _date = datetime.strptime(dcm.AcquisitionDate, '%Y%m%d')
            date = '{:02d}.{:02d}.{:d}'.format(
                _date.day,
                _date.month,
                _date.year,
            )
            meta['Acquisition Date'] = str(date)

        if hasattr(dcm, 'AcquisitionTime'):
            _time = datetime.strptime(dcm.AcquisitionTime, '%H%M%S.%f')
            time = '{:02d}:{:02d}:{:02d}'.format(
                _time.hour,
                _time.minute,
                _time.second,
            )
            meta['Acquisition Time'] = str(time)

        if hasattr(dcm, 'PatientName'):
            meta['Patient Name'] = str(dcm.PatientName)

        if hasattr(dcm, 'PatientSex'):
            meta['Patient Sex'] = str(dcm.PatientSex)

        if hasattr(dcm, 'BodyPartExamined'):
            meta['Body Part'] = str(dcm.BodyPartExamined)

        if hasattr(dcm, 'PerformingPhysicianName'):
            meta['Physician'] = str(dcm.PerformingPhysicianName)

        if hasattr(dcm, 'InstitutionName'):
            meta['Institution'] = str(dcm.InstitutionName)

        if hasattr(dcm, 'Manufacturer'):
            meta['Manufacturer'] = str(dcm.Manufacturer)

        if hasattr(dcm, 'Modality'):
            meta['Modality'] = str(dcm.Modality)

        if hasattr(dcm, 'ImageType'):
            meta['Image Type'] = str(dcm.ImageType)

        meta['Slices'] = dcm.pixel_array.shape[0]
        meta['Height'] = dcm.pixel_array.shape[1]
        meta['Width'] = dcm.pixel_array.shape[2]
        meta['Channels'] = dcm.pixel_array.shape[3]
        meta['Data Type'] = dcm.pixel_array.dtype

        if hasattr(dcm, 'WindowCenter'):
            meta['WC'] = int(float(dcm.WindowCenter))

        if hasattr(dcm, 'WindowWidth'):
            meta['WW'] = int(float(dcm.WindowWidth))

        log.info(f'Processed DICOM: {dcm_path}')

    except Exception as e:
        log.warning(f'Broken DICOM: {dcm_path}')

    return meta


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='get_study_metadata',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    dcm_list = get_file_list(
        src_dirs=cfg.data_dir,
        ext_list='',
        filename_template='IMG',
    )

    meta = []
    for dcm_path in tqdm(dcm_list, desc='Extract metadata', unit=' dicoms'):
        dcm_meta = extract_metadata(
            dcm_path=dcm_path,
        )
        meta.append(dcm_meta)

    df = pd.DataFrame(meta)
    df.sort_values(by='Path')
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, 'metadata.xlsx')
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )

    log.info(f'Metadata saved: {save_path}')


if __name__ == '__main__':
    main()
