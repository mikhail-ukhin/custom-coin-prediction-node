import os

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    else:
        print("The file does not exist.")

def remove_files_in_dir(path):
    for filename in os.listdir(path):
        print(filename)
        file_path = os.path.join(path, filename)
        
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")