import hashlib
import os

from clearml import Dataset

from src.data.utils import get_file_list


class DataManager:
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
            hash_local = (
                self.compute_dir_hash(dir_path=self.data_dir)
                if num_local_images > 0
                else float('nan')
            )
        except Exception:
            is_local_exist = False
            num_local_images = 0
            hash_local = float('nan')

        # Check remote dataset
        try:
            dataset = Dataset.get(
                dataset_name=self.dataset_name,
                dataset_project=self.project_name,
                only_completed=True,
            )
            hash_remote = dataset.tags[0]
            is_remote_exist = True
        except Exception:
            is_remote_exist = False
            hash_remote = float('nan')

        # Raise an error if both datasets do not exist
        assert is_local_exist or is_remote_exist, 'Neither local nor remote dataset exists'

        if num_local_images != 0 and (not is_remote_exist or hash_local != hash_remote):
            self.upload_dataset(hash_value=hash_local)
        elif hash_local == hash_remote:
            print('\nDatasets on both sides are completely consistent\n')
        else:
            self.download_dataset(dataset=dataset)

    def upload_dataset(
        self,
        hash_value: str,
    ):
        # Check subsets
        train_pairs = get_file_list(
            src_dirs=os.path.join(self.data_dir, 'train'),
            ext_list='.png',
        )
        test_pairs = get_file_list(
            src_dirs=os.path.join(self.data_dir, 'test'),
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

        # Create a dataset instance and add files to it
        print('Uploading dataset...\n')
        dataset = Dataset.create(
            dataset_name=self.dataset_name,
            dataset_project=self.project_name,
        )
        dataset.add_files(
            path=self.data_dir,
            verbose=True,
        )

        # Log dataset statistics
        dataset.get_logger().report_histogram(
            title='Dataset split',
            series='Images',
            xaxis='Subset',
            yaxis='Images',
            xlabels=['train', 'test'],
            values=[num_train_images, num_test_images],
        )

        # Upload dataset
        dataset.add_tags(hash_value)
        dataset.upload(
            show_progress=True,
            verbose=True,
        )
        dataset.finalize(verbose=True)
        print('Upload complete')

    def download_dataset(
        self,
        dataset: Dataset,
    ):
        num_images = self.count_images(file_paths=dataset.list_files())
        num_pairs = num_images // 2
        print(f'Downloading a dataset of {num_pairs} pairs')
        local_path = dataset.get_mutable_local_copy(
            target_folder=self.data_dir,
        )
        print(f'Dataset downloaded: {local_path}')

    @staticmethod
    def compute_dir_hash(
        dir_path: str,
        length: int = 8,
    ):
        sha_hash = hashlib.sha256()
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        sha_hash.update(chunk)

        hash_value_ = sha_hash.hexdigest()
        hash_value = hash_value_ if 0 > length > 64 else hash_value_[:length]
        return hash_value

    @staticmethod
    def count_images(file_paths):
        image_extensions = [
            '.jpg',
            '.jpeg',
            '.png',
            '.bmp',
        ]
        image_count = 0

        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in image_extensions:
                image_count += 1

        return image_count


if __name__ == '__main__':
    processor = DataManager(
        data_dir='data/final',
        project_name='OCT segmentation',
    )
    hash_value = processor.compute_dir_hash('data/final')
    processor.prepare_data()
    print('Complete')
