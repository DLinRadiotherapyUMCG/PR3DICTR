import os
import sqlite3


def create_file(pathFile):
    dirPath = os.path.dirname(pathFile)

    if os.path.exists(dirPath) == False:
        create_folder(dirPath)

    if os.path.exists(pathFile) == False:
        if(pathFile.endswith('.db')):
            create_database(pathFile)     
        else:
            f = open(pathFile,"a")
            f.close()      

def create_folder(path):
    dirPath = path
    # Check is file
    if os.path.isfile(dirPath):
        dirPath = os.path.dirname(dirPath)

    # Check if it ends with /
    if dirPath.endswith(os.path.sep) or dirPath.endswith('\\') or dirPath.endswith('/'):
        dirPath = dirPath[:-1]

    # Now check to create directories accordingly
    if os.path.exists(dirPath) == False:
        os.makedirs(dirPath, exist_ok = True)

def create_database(pathFile):
    conn = None
    try:
        conn = sqlite3.connect(pathFile)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def create_textfile(pathGiven, pathName, text):
    pathDirectory = str(pathGiven)
    if os.path.exists(pathDirectory) == False:
        create_folder(pathDirectory)

    if pathDirectory.endswith(os.path.sep) or pathDirectory.endswith('\\') or pathDirectory.endswith('/'):
        pathDirectory = pathDirectory[:-1]

    if(pathName.endswith('.txt')):
        pathName = pathName[:-4]
    
    for i in range(1000):
        if i == 0:
            pathFile = os.path.join(pathDirectory,pathName + ".txt")
        else:
            pathFile = os.path.join(pathDirectory,pathName + f"{i}".rjust(3,"0") + ".txt")
        if os.path.exists(pathFile) == False:
            f = open(pathFile,"a", encoding='utf-8')
            f.write(text)
            f.close()
            break

        

    