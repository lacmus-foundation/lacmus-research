import os
from os.path import exists
import requests
import shutil
import tarfile

data_dir = '../data/laddv4'

gdrive_file_ids = {
    'Winter': '127IYHwO57tdKkP3143qLLZHDxRFg8urJ',
    'Summer': '18EDxcsGY_azTQAE8cVxJluBq3SgLEhbi',
    'Spring': '1r94U4wgQXGGX2xcvgcBTDP30Vj7aEvpd'
}

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    
    
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def prepare_season_dataset(season, file_id, prefix='LADDV4_'):
    dataset = prefix + season
    dataset_dir = data_dir + '/' + dataset
    if not exists(dataset_dir):
        archive = dataset + '.tar.xz' 
        archive_path = data_dir + '/' + archive
        if not exists(archive_path):
            download_file_from_google_drive(file_id, archive_path)
    
        season_archive = tarfile.open(archive_path)
        season_archive.extractall(data_dir)
        season_archive.close()
        os.rename(os.path.join(data_dir, dataset), os.path.join(data_dir, season.lower()))


def copy_files(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)


def merge_season_datasets(merged_name='full'):
    merged_dir = os.path.join(data_dir, merged_name)
    os.mkdir(merged_dir)
    merged_images_dir = os.path.join(merged_dir, 'JPEGImages')
    os.mkdir(merged_images_dir)
    merged_annotations_dir = os.path.join(merged_dir, 'Annotations')
    os.mkdir(merged_annotations_dir)
    merged_sets_dir = os.path.join(merged_dir, 'ImageSets')
    os.mkdir(merged_sets_dir)
    merged_sets_dir = os.path.join(merged_sets_dir, 'Main')
    os.mkdir(merged_sets_dir)

    for season in gdrive_file_ids.keys():
        season = season.lower()
        season_dir = os.path.join(data_dir, season)

        season_images_dir = os.path.join(season_dir, 'JPEGImages')
        copy_files(season_images_dir, merged_images_dir)

        season_annotations_dir = os.path.join(season_dir, 'Annotations')
        copy_files(season_annotations_dir, merged_annotations_dir)

        season_sets_dir = os.path.join(season_dir, 'ImageSets', 'Main')
        for set_name in ['test.txt', 'train.txt', 'trainval.txt', 'val.txt']:
            season_set_file = os.path.join(season_sets_dir, set_name)
            merget_set_file = os.path.join(merged_sets_dir, set_name)
            with open(merget_set_file, "a") as merged_file:
                with open(season_set_file, 'r') as season_file:
                    merged_file.writelines(season_file.readlines())


if __name__ == '__main__':
    for season in gdrive_file_ids:
        prepare_season_dataset(season, gdrive_file_ids[season])

    merge_season_datasets()

    



