import os

def write_points_to_file(file_path, data):
    """
    Write (x,y) points to the specified file path
    :param file_path: Full path to the file
    :param data: Data in format [(x1, y1), ... ,(xn, yn)]
    :return: Success status
    """
    f = open(file_path,'w')
    for d in data:
        if data.index(d) == 0:
            f.write("x,y\n")
        f.write(f"{str(d[0])},{str(d[1])}\n")
    f.close()