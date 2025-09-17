import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

height, width = 256, 256


# It is important to add this line if you are using the UV Package manager
matplotlib.use("Qt5Agg")  # or 'Qt5Agg' if using PyQt5


def main():
    data = np.load("data/Maps_B_IllustrisTNG_1P_z=0.00.npy")
    nframes = len(data)
    filename = "visualization.mp4"
    fps = 5
    height, width = data.shape[1], data.shape[2]

    min_val = np.min(data)
    max_val = np.max(data)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "gray",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "rgb24",
            f"{filename}",
        ],
        stdin=subprocess.PIPE,
    )

    for i in tqdm(range(nframes), desc="Processing video...", unit="frames", total=nframes):
        _data_flatten = data[i].flatten()
        # TODO: Fix to be logarithmic scaling. Base 10. B, Mcdm, Mtot,
        # TODO: Look at correlation + Power Spectrum
        _data_flatten = np.log(_data_flatten)
        _data = (_data_flatten - _data_flatten.min()) / (_data_flatten.max() - _data_flatten.min())
        _data = _data.reshape(data[i].shape)
        frame = (_data * 255).astype(np.uint8)
        ffmpeg.stdin.write(frame.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()


if __name__ == "__main__":
    main()
