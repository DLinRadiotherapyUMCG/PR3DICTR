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
    if(os.path.isdir(dirPath) == False):
        dirPath = os.path.dirname(dirPath)
    if(os.path.exists(dirPath) == False):
        os.makedirs(dirPath)

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
        

    