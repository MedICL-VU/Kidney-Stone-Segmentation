"""
Video [class]
Inits: 
code: specify the fourcc code for the output video. Will vary based on your operating system. 
ext: specify the output video file extension. Should correspond to the fourcc code.  
Refer to this website https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html. 

apply(): process every frame a video and save the output as a new video
"""

import os
import cv2
import torch
import math
import torch.nn.functional as F
from abc import ABC
from util import create_sliding_window_tensor, recon_and_rectify, write_video

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_frames(vid):
    vid_frames = []
    while (1):
        success, frame = vid.read()
        if success:
            vid_frames.append(frame)
        else:
            break

    return vid_frames


class Video:
    def __init__(self, path, code='MJPG', ext='.mp4'):
        self.path = path
        self.code = code
        self.ext = ext
        self.cap = cv2.VideoCapture(path)
        self.name, self.root = self.__name_root()
        # https://stackoverflow.com/questions/61723675/crop-a-video-in-python
        # video attributes
        self.w_frame, self.h_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps, self.frames = self.cap.get(cv2.CAP_PROP_FPS), self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # helper to acquire name and root from the video's path
    def __name_root(self):
        root = os.path.splitext(self.path)[0]
        split = root.split('/')
        return split[-1], root

    # apply an arbitrary function to the video
    # save output video and transformed stills in new directory
    def apply(self, frequency, fnc, **kwargs):
        cap = self.cap

        # output
        out_dir = self.root + '_' + fnc.__name__
        out_name = self.name + '_' + fnc.__name__
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # fourcc = cv2.VideoWriter_fourcc(*self.code)
        # out = cv2.VideoWriter(os.path.join(out_dir,out_name+self.ext), fourcc, self.fps, (640, 480))

        count = 0
        out_frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                transformed = fnc(frame, **kwargs)
                resized = cv2.resize(transformed, (448, 448))
                out_frames += [resized]

                if count % frequency == 0:
                    cv2.imwrite(os.path.join(out_dir, out_name + '_' + str(int(count / frequency) + 1) + '.jpg'),
                                resized)

                # out.write(resized)
                count += 1

                cv2.imshow('original', cv2.resize(frame, (448, 448)))
                cv2.imshow('transformed', resized)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        write_video(os.path.join(out_dir, out_name + '.mp4'), out_frames, self.fps, (448, 448))
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


################################
### VIDEO SEQUENCE INPUT HELPERS
################################

# abstract base class
class VideoFeature(ABC):
    def __init__(self, vid):
        self.vid = vid
        self.seg = torch.zeros(self.vid.size())

    def getFeature(self, n):
        pass

    def setFeature(self, n, f):
        pass


# sliding window extraction
class SlidingWindow(VideoFeature):
    def __init__(self, vid, w):
        super(SlidingWindow, self).__init__(vid)
        self.v_sz = self.vid.size()
        self.w = w
        self.p_sz = (*self.v_sz[:3], w)
        self.s_sz = (*self.v_sz[:3], 1)
        self.windows = torch.from_numpy(create_sliding_window_tensor(self.vid.numpy(), self.p_sz, self.s_sz))
        self.w_sz = self.windows.size()
        self.seg = torch.zeros(self.w_sz[0], 1, *self.w_sz[2:])

    def __iter__(self):
        for window in self.windows:
            yield window

    def getFeature(self, n):
        return self.windows[n]

    def setFeature(self, n, f):
        self.seg[n] = f

    def getRecon(self):
        return recon_and_rectify(self.seg, tuple(self.v_sz), self.w)
