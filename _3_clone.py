"""
This script is used to prepare the data for training YOLOv8:

Clone the folder for 4 times for multi-threading
"""
import os
from dotenv import load_dotenv

load_dotenv(".env")
DIR_DATA_STUDY1 = os.getenv("DIR_DATA_STUDY1")
DIR_DATA_STUDY2 = os.getenv("DIR_DATA_STUDY2")

def main():
    for i in range(4):
        os.system("cp -r %s %s_%d" % (DIR_DATA_STUDY1, DIR_DATA_STUDY1, i))
        os.system("cp -r %s %s_%d" % (DIR_DATA_STUDY2, DIR_DATA_STUDY2, i))
        
if __name__ == "__main__":
    main()
