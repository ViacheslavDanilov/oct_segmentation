import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import pydicom
from tqdm import tqdm

from tools.utils import get_file_list

logger = logging.getLogger(__name__)


def extract_metadata(
        dcm_path: str,
) -> dict:

    dcm = pydicom.dcmread(dcm_path)
    keys = [
        'Path',
        'Study UID',
        'Series UID',
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

    logger.info('Processed DICOM: {:s}'.format(dcm_path))

    return meta


if __name__ == '__main__':

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d.%m.%Y %I:%M:%S',
        filename='logs/{:s}.log'.format(Path(__file__).stem),
        filemode='w',
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description='Retrieve and save DICOM metadata')
    parser.add_argument('--study_dir', required=True, type=str, help='directory with studies')
    parser.add_argument('--save_dir', required=True, type=str, help='directory where to save CSV file')
    args = parser.parse_args()

    # Include or exclude specific directories
    dcm_list = get_file_list(
        src_dirs=args.study_dir,
        ext_list='',
        filename_template='IMG',
    )

    meta = []
    for dcm_path in tqdm(dcm_list, desc='Extract metadata', unit=' dicoms'):
        dcm_meta = extract_metadata(
            dcm_path=dcm_path
        )
        meta.append(dcm_meta)

    df = pd.DataFrame(meta)
    df.sort_values(by='Path')
    save_path = os.path.join(args.save_dir, 'meta.xlsx')
    df.to_excel(save_path, sheet_name='Meta', index=False, startrow=0, startcol=0)

    logger.info('Metadata saved: {:s}'.format(save_path))
