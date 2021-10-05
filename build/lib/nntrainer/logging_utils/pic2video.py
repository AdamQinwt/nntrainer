import os
import cv2



def convert(file_list,save_path,fmt='mp4',img_size = (1280, 720),fps=12):
    if fmt=='mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif fmt=='avi':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, img_size)
    for file_name in file_list:
        try:
            img = cv2.imread(file_name)
            video_writer.write(img)
        except:
            print(f'Unable to load {file_name}!')
            break
    video_writer.release()