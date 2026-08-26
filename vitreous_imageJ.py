import os, cv2
from PIL import Image
from PIL.TiffTags import TAGS


def open_pil(fpath):
    # https://stackoverflow.com/questions/46477712/reading-tiff-image-metadata-in-python
    with Image.open(fpath) as img:
        meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.iterkeys()}
    return


def open_pil_v2(fpath):
    # https://stackoverflow.com/questions/55040017/read-tiff-tags-in-python
    # https://stackoverflow.com/questions/20529187/what-is-the-best-way-to-save-image-metadata-alongside-a-tif

    return


def open_imageJ(fpath):
    # https://imagej.nih.gov/ij/docs/menus/file.html#saveas
    # https://docs.oracle.com/javase%2F9%2Fdocs%2Fapi%2F%2F/javax/imageio/metadata/doc-files/tiff_metadata.html
    # https://imagej.net/scripting/python

    # https://py.imagej.net/en/latest/06-Working-with-Images.html#opening-images-with-ij-io-open
    import imagej
    ij = imagej.init(mode='interactive')
    print("ImageJ2 version: {}".format(ij.getVersion()))
    dataset = ij.io().open(fpath)
    ij.py.show(dataset)
    imp = ij.IJ.openImage(fpath)
    return


def open_tifffile(fpath):
    import tifffile
    with tifffile.TiffFile(fpath) as tif:
        data = tif.asarray()
        ## metadata = tif[0].image_description

        # https://docs.python.org/3/library/stdtypes.html
        # https://www.geeksforgeeks.org/how-to-convert-bytes-to-int-in-python/
        temp = int.from_bytes(tif.imagej_metadata["Overlays"], "big")
        print(fpath, len(tif.imagej_metadata["Overlays"]), len(str(temp)))
        # int.from_bytes(tif.imagej_metadata["Overlays"], "little")
        # int.from_bytes(tif.imagej_metadata["Overlays"], "big", signed=True)
    return


if __name__ == '__main__':
    imageJ_folder = os.path.join("vitreous", "ImageJ_Comparisons")
    raw_folder = os.path.join(imageJ_folder, "raw")
    fnames = os.listdir(imageJ_folder)
    for fdx, fname in enumerate(fnames):
        fpath = os.path.join(imageJ_folder, fname)
        # open_pil(fpath)
        # open_imageJ(fpath)
        open_tifffile(fpath)
