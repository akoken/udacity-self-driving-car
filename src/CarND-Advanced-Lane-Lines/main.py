from lane import Lane
from moviepy.editor import VideoFileClip

def pipeline(img):
    # Find the lane lines first
    img_lanes = lane.detect_lines(img)
    # Return image with found cars and lanes
    return img_lanes

if __name__=='__main__':
    # Create lane object
    lane = Lane()
    selector = "project"
    output_video = 'out_{}_video.mp4'.format(selector)
    clip = VideoFileClip('{}_video.mp4'.format(selector))
    video_clip = clip.fl_image(pipeline)
    video_clip.write_videofile(output_video, audio=False)
