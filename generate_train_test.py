import glob

from sklearn.model_selection import train_test_split

from utils.iotools import mkdir_if_missing
from Settings import Settings


def write_in_file(file_path, data):
    with open(file_path,mode='w') as file:
        for d in data:
            file.write(str(d) + '\n')
    return True



if __name__ == "__main__":
    mkdir_if_missing(Settings.SPLIT_PATH)
    train_path = f'{Settings.SPLIT_PATH}train.txt'
    val_path = f'{Settings.SPLIT_PATH}val.txt'
    data = tuple(f for f in glob.glob(f'{Settings.DATASET_PATH}*/*'))
    train, val = train_test_split(data, test_size=0.2)
    write_in_file(train_path, train)
    write_in_file(val_path, val)