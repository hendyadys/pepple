import os, json, csv
import io
from googleapiclient.http import MediaIoBaseDownload
from sys import platform

if platform == "win32":
    base_folder = os.path.join("z:/", "yue", "pepple")
else:
    base_folder = os.path.join("/data", "yue", "pepple")
data_folder = os.path.join("data_2024")
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)
save_folder = os.path.join(data_folder, "img_folder")
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)


def parse_csv_file(csv_file, id_name="eaid"):
    demo_dict = {}
    counter = 0
    with open(csv_file, "r") as fin:
        freader = csv.reader(fin, delimiter=",", quotechar='"')     # robust to ","
        for row in freader:
            counter +=1
            if counter==1:
                header = [x.strip() for x in row]
            else:
                cur_dict = dict(zip(header, row))
                id_field = cur_dict[id_name]

                if id_field not in demo_dict:
                    demo_dict[id_field] = [cur_dict]
                else:
                    demo_dict[id_field].append(cur_dict)
    return demo_dict


def parse_data_file():
    # data_file = os.path.join(data_folder, "final_score_original.csv")
    # data_file = os.path.join(data_folder, "final_score_textOnly.csv")
    data_file = os.path.join(data_folder, "final_score_textOnly_new.csv")   # new as of 2025-03-21
    data_file = os.path.join(data_folder, "all_tiffs_20250402.csv")   # new as of 2025-03-21
    data_file = os.path.join(data_folder, "all_tiffs_20250402_fixed.csv")   # new as of 2025-04-10: fixed urls
    img_dict = parse_csv_file(data_file, id_name="TIFF Name")

    corrected_AC_file = os.path.join(data_folder, "corrections_AC.csv")
    corrected_AC_dict = parse_csv_file(corrected_AC_file, id_name="Tiff")
    img_dict= adjust_score(img_dict, corrected_AC_dict)
    # check intersection
    img_dict_keys = list(img_dict.keys())
    corrected_AC_dict_keys = [x.replace(".TIFF", "") for x in list(corrected_AC_dict.keys())]
    keys_to_be_corrected_AC = set(img_dict_keys).intersection(corrected_AC_dict_keys)

    corrected_PC_file = os.path.join(data_folder, "corrections_PC.csv")
    corrected_PC_dict = parse_csv_file(corrected_PC_file, id_name="Tiff")
    img_dict = adjust_score(img_dict, corrected_PC_dict)
    # check intersection
    corrected_PC_dict_keys = [x.replace(".TIFF", "") for x in list(corrected_PC_dict.keys())]
    keys_to_be_corrected_PC = set(img_dict_keys).intersection(corrected_PC_dict_keys)

    # get images
    service = get_google_service()
    counter = 0
    for tiff_name, tiff_dict in img_dict.items():
        tiff_data = tiff_dict[0]
        tiff_name = tiff_data["TIFF Name"]
        if ".tiff" not in tiff_name.lower():
            tiff_path = os.path.join(save_folder, "{}.tiff".format(tiff_name))
        else:
            tiff_path = os.path.join(save_folder, "{}".format(tiff_name))
        if not os.path.isfile(tiff_path):
            counter +=1
            print(counter, "grabbing new file:", tiff_path)
            get_google_img(service, tiff_data["Link to Image"], tiff_path)
    return img_dict


def adjust_score(img_dict, corrected_scores_dict):
    for tiff_name, tiff_dict in corrected_scores_dict.items():
        if tiff_name.replace(".TIFF", "") in img_dict:
            img_dict[tiff_name]["score"] = tiff_dict["Score"]
    return img_dict


# could also use rclone
def get_google_img(service, url, tiff_path):
    # eg https://drive.google.com/file/d/16tNrWXMBbGIJLVTsPcJhp7jgVACRNegl/view?usp=sharing
    url_toks = url.split("/")
    file_id = url_toks[-2]
    download_file(service, file_id, tiff_path)
    return


def download_file(service, file_id, file_name):
    """Download a file from Google Drive."""
    try:
        # Request the file from Google Drive
        request = service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()

        # Use MediaIoBaseDownload to download the file
        downloader = MediaIoBaseDownload(file_handle, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

        # Save the file to the local filesystem
        with open(file_name, "wb") as f:
            f.write(file_handle.getvalue())
        print(f"File downloaded successfully: {file_name}")

    except Exception as e:
        print(f"An error occurred: {e}")
    return


def get_google_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    
    # Authenticate with your credentials.json
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(os.path.join(data_folder, 'credentials.json'), scopes=SCOPES)

    # Connect to Google Drive API
    service = build('drive', 'v3', credentials=creds)
    return service


if __name__ == '__main__':
    parse_data_file()