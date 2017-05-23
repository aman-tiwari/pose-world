from __future__ import print_function

import numpy as np
import cv2

from infer import handle_one

if __name__ == "__main__":
    print('warming up')
    _ = handle_one(np.ones((1, 320, 320, 3)))

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        canvas = handle_one(frame)

        # Display the resulting frame
        cv2.imshow('Video', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
