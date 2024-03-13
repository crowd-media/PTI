import logging
import subprocess

import numpy as np

from unith_thai.helpers.writer.frame_writer import FrameWriter

logger = logging.getLogger(__name__)


class FfmpegVideoWriter(FrameWriter):
    def __init__(
        self,
        output_video_path: str,
        video_height: int,
        video_width: int,
        fps: int,
        start_frame: int = 0,
    ):
        """
        :param output_video_path: Path for output video.
        :param audio_input_path: Path to input audio.
        :param start_frame: Start reading from this frame of the input video.
        :param video_width: The width of the output video
        :param video_height: The height of the output video
        :param fps: The frames per second of the output video
        """
        self.output_video_path = output_video_path
        self.start_frame = start_frame
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.ffmpeg_process = None
        super().__init__()

    def _start_ffmpeg_process(self) -> subprocess.Popen:
        # Build the FFmpeg command
        ffmpeg_command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-loglevel",
            "error",  # Log level
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.video_width}x{self.video_height}",  # size of one frame
            "-pix_fmt",
            "bgr24",
            "-r",
            str(self.fps),  # Frame rate
            "-i",
            "-",  # Input from pipe
            "-preset",
            "ultrafast",
            "-af",
            "adelay=%s" % (1000 * self.start_frame / self.fps),
            "-acodec",
            "aac",
            "-ar",
            "44100",
            "-pix_fmt",
            "yuv420p",
            self.output_video_path,
        ]

        # Open the subprocess with the FFmpeg command
        return subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    def write_frame(self, frame: np.ndarray) -> None:
        frame = frame.ravel()
        self.ffmpeg_process.stdin.write(frame.data)
        self.ffmpeg_process.stdin.flush()

    def start(self) -> None:
        self.ffmpeg_process = self._start_ffmpeg_process()

    def stop(self) -> None:
        if self.ffmpeg_process:
            logger.info("Closing pipe")
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
            logger.info("Video write finished")
