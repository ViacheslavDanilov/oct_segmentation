import hashlib
import os

from clearml import Dataset

from src.data.utils import get_file_list


class ClearMLDataset:
    """A class for processing data in the ClearML framework."""

    def __init__(
        self,
        data_dir: str,
        project_name: str = 'OCT segmentation',
    ):
        self.data_dir = data_dir
        self.dataset_name = os.path.basename(data_dir)
        self.project_name = project_name

    def prepare_data(self):
        # Check local dataset
        try:
            local_image_paths = get_file_list(
                src_dirs=self.data_dir,
                ext_list='.png',
            )
            num_local_images = len(local_image_paths)
            is_local_exist = True if num_local_images > 0 else False
            hash_local = self.compute_dir_hash(dir_path=self.data_dir)
        except Exception:
            is_local_exist = False
            num_local_images = 0
            hash_local = float('nan')

        # Check remote dataset
        try:
            clearml_dataset = Dataset.get(
                dataset_name=self.dataset_name,
                dataset_project=self.project_name,
                # dataset_version='1.0.0',
                only_completed=True,
            )
            num_removed_files, num_added_files, num_modified_files = clearml_dataset.sync_folder(
                local_path=self.data_dir,
                verbose=True,
            )
            hash_remote = clearml_dataset.tags[0]
            is_remote_exist = True
        except Exception:
            is_remote_exist = False
            num_removed_files, num_added_files, num_modified_files = 0, 0, 0
            hash_remote = float('nan')

        # Raise an error if both datasets do not exist
        assert is_local_exist or is_remote_exist, 'Neither local nor remote dataset exists'

        # TODO: Complete
        if (
            sum([num_removed_files, num_added_files, num_modified_files]) > 1
            and num_local_images != 0
        ):
            step = 'upload'
        elif hash_local == hash_remote:
            step = 'nothing'
        else:
            step = 'download'

        # Run one of three possible scenarios
        if step == 'upload':
            self.upload_dataset()
        elif step == 'download':
            self.download_dataset()
        elif step == 'nothing':
            print('Datasets on both sides are completely consistent')
        else:
            raise ValueError('Unknown step')

    def upload_dataset(self):
        print('Uploading dataset...')

        # Create a dataset instance and add files to it
        dataset = Dataset.create(
            dataset_name=self.dataset_name,
            dataset_project=self.project_name,
            # dataset_version='1.0.0',
        )
        dataset.add_files(
            path=self.data_dir,
            verbose=True,
        )

        # Check subsets
        train_pairs = get_file_list(
            src_dirs=os.path.join(self.data_dir, 'train'),
            ext_list='.png',
        )
        test_pairs = get_file_list(
            src_dirs=os.path.join(self.data_dir, 'train'),
            ext_list='.png',
        )
        if len(train_pairs) % 2 != 0:
            raise ValueError('Inconsistent number of train images and masks.')
        else:
            num_train_images = len(train_pairs) // 2

        if len(test_pairs) % 2 != 0:
            raise ValueError('Inconsistent number of test images and masks.')
        else:
            num_test_images = len(test_pairs) // 2

        # Log dataset statistics
        dataset.get_logger().report_histogram(
            title='Dataset split',
            series='Images',
            xlabels=['train', 'test'],
            values=[num_train_images, num_test_images],
        )

        # Upload dataset
        # TODO: set hash tag to the dataset
        hash_value = self.compute_dir_hash(self.data_dir)
        dataset.add_tags(hash_value)
        dataset.upload(
            show_progress=True,
            verbose=True,
        )
        dataset.finalize(verbose=True)
        print('Complete')

    @staticmethod
    def download_dataset():
        print('Downloading...')
        print('Downloading...')

    @staticmethod
    def compute_dir_hash(
        dir_path,
    ):
        sha_hash = hashlib.sha256()
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        sha_hash.update(chunk)
        return sha_hash.hexdigest()


if __name__ == '__main__':
    processor = ClearMLDataset(
        data_dir='data/final',
        project_name='OCT segmentation',
    )
    processor.prepare_data()
    print('Complete')
