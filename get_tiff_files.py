import os


def get_files_in_folder(folder, ext='.tiff'):
    all_files = os.listdir(folder)
    files = [f for f in all_files if ext in f]
    return files


if __name__ == '__main__':
    folder = '.'
    get_files_in_folder(folder)