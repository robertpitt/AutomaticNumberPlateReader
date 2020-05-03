import cv2
import numpy as np
import youtube_dl
from vidstab.VidStab import VidStab
import matplotlib.pyplot as plt

stabilizer = VidStab()

if __name__ == '__main__':

    video_url = 'https://youtu.be/-drIOjUXkPs'

    ydl_opts = {}

    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)

    # get video formats available
    entries = info_dict.get('entries',None)

    for f in entries[0].get("formats"):
        # I want the lowest resolution, so I set resolution as 144p
        if f.get('height',None) == 480: # 240, 360, 480, 720, 1080

            #get the video url
            url = f.get('url',None)

            # open url with opencv
            cap = cv2.VideoCapture(url)

            # check if url was opened
            if not cap.isOpened():
                print('video not opened')
                exit(-1)            

            while True:
                # read frame
                ret, frame = cap.read()

                # check if frame is empty
                if not ret:
                    break

                # Pass frame to stabilizer even if frame is None
                # stabilized_frame will be an all black frame until iteration 30
                stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=80)
                if stabilized_frame is None:
                  # There are no more frames available to stabilize
                  break

                # cv2.calc

                # Mat M = estimateRigidTransform(frame1,frame2,0)
                # warpAffine(frame2,output,M,Size(640,480),INTER_NEAREST|WARP_INVERSE_MAP) 

                # display frame
                cv2.imshow('frame', frame)
                cv2.imshow('stabilized_frame', stabilized_frame)

                # stabilizer.plot_trajectory()
                # plt.show()

                stabilizer.plot_transforms()
                plt.show()

                if cv2.waitKey(30)&0xFF == ord('q'):
                    break

            # release VideoCapture
            cap.release()

    cv2.destroyAllWindows()