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
        # TODO: if local data is newer than server data, upload it

        clearml_dataset = Dataset.get(
            dataset_name=self.dataset_name,
            dataset_project=self.project_name,
            only_completed=True,
        )
        num_removed_files, num_added_files, num_modified_files = clearml_dataset.sync_folder(
            local_path=self.data_dir,
            verbose=True,
        )
        if sum([num_removed_files, num_added_files, num_modified_files]) > 1:
            dataset = Dataset.create(
                dataset_name=self.dataset_name,
                dataset_project=self.project_name,
            )
            self.upload_dataset(dataset)

        # a = clearml_dataset.verify_dataset_hash(
        #     local_copy_path=self.data_dir,
        #     skip_hash=True,
        #     verbose=True,
        # )

        # TODO: if no data is available locally, download it from the server
        self.download_dataset()

    def upload_dataset(
        self,
        dataset: Dataset,
    ):
        # Create a dataset instance
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

    def compute_dir_hash(
        self,
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
