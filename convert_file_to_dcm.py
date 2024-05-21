import os
import shutil

def convert_files_to_dcm(input_folder, output_folder):
    
    # Process all files in the specified folder
    for filename in os.listdir(input_folder):
        
        # Create file path
        file_path = os.path.join(input_folder, filename)
        
        # If there is no “.dcm” extension in the file name and the file type is “File”
        if not filename.endswith(".dcm") and os.path.isfile(file_path):
            
            # Create the converted filename by appending the “.dcm” extension from the filename
            new_filename = filename + ".dcm"
            new_file_path = os.path.join(output_folder, new_filename)
            
            # If the new filename is not available, convert the file
            if not os.path.exists(new_file_path):
                
                # Copy file with new name
                shutil.copy(file_path, new_file_path)
                print(f"{filename} is converted as {new_filename}.")
            else:
                print(f"A file named {new_filename} already exists.")
        else:
            print(f"{filename} file not processed.")

# Input and output directory paths
input_folder = r"your\input\directory\path"
output_folder = r"your\output\directory\path"

# Call the convert function
convert_files_to_dcm(input_folder, output_folder)
