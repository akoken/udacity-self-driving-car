from lane import Lane
from moviepy.editor import VideoFileClip

def pipeline(img):
    img_lanes = lane.detect_lines(img)
    return img_lanes

if __name__=='__main__':
    lane = Lane()
    selector = "project"
    output_video = 'out_{}_video.mp4'.format(selector)
    clip = VideoFileClip('{}_video.mp4'.format(selector))
    video_clip = clip.fl_image(pipeline)
    video_clip.write_videofile(output_video, audio=False)
