import os
import sqlite3


def create_file(pathFile):
    dirPath = os.path.dirname(pathFile)
    if(os.path.exists(dirPath) == False):
        os.mkdir(dirPath)
    if(os.path.exists(pathFile) == False):
        if(pathFile.endswith('.db')):
            create_database(pathFile)            

def create_folder(path):
    dirPath = path
    # Check is file
    if(os.path.isdir(dirPath) == False):
        dirPath = os.path.dirname(dirPath)

    # Check if it ends with /
    if(dirPath.endswith(os.path.sep) == False):
        dirPath += os.path.sep

    # Now check to create directories accordingly
    if(os.path.exists(dirPath) == False):
        os.makedirs(dirPath, exist_ok = True)

def create_database(pathFile):
    conn = None
    try:
        conn = sqlite3.connect(pathFile)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if(conn):
            conn.close()

def create_textfile(pathDirectory, pathName, text):
    if(os.path.exists(pathDirectory) == False):
        create_folder(pathDirectory)
    for i in range(1000):
        if(i == 0):
            pathFile = os.path.join(pathDirectory,pathName + ".txt")
        else:
            pathFile = os.path.join(pathDirectory,pathName + f"{i}".rjust(3,"0") + ".txt")
        if(os.path.exists(pathFile) == False):
            f = open(pathFile,"a")
            f.write(text)
            f.close()
            break

        

    