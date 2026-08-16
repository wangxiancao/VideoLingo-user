import os
import shutil

def delete_dubbing_files():
    files_to_delete = [
        os.path.join("output", "dub.wav"),
        os.path.join("output", "output_dub.mp4"),
        # The merged dub track (written by _11, consumed by _12) and the
        # volume-normalized copy produced by _12, so a failed retry can never
        # let _12 consume stale dub audio.
        os.path.join("output", "dub.mp3"),
        os.path.join("output", "normalized_dub.wav"),
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    
    segs_folder = os.path.join("output", "audio", "segs")
    if os.path.exists(segs_folder):
        try:
            shutil.rmtree(segs_folder)
            print(f"Deleted folder and contents: {segs_folder}")
        except Exception as e:
            print(f"Error deleting folder {segs_folder}: {str(e)}")
    else:
        print(f"Folder not found: {segs_folder}")

    # Also clear the temp TTS audio directory so retries cannot reuse stale
    # pre-speed audio from a previous run.
    tmp_folder = os.path.join("output", "audio", "tmp")
    if os.path.exists(tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
            print(f"Deleted folder and contents: {tmp_folder}")
        except Exception as e:
            print(f"Error deleting folder {tmp_folder}: {str(e)}")
    else:
        print(f"Folder not found: {tmp_folder}")

if __name__ == "__main__":
    delete_dubbing_files()