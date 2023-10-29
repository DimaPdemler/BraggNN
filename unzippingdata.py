import tarfile

with tarfile.open('dataset/dataset.tar.gz', 'r:gz') as tar:
    tar.extractall(path='unzipped_data')
