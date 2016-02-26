#!/usr/bin/env python2
# A script which downloads the selected list of files from YouTube, then converts them into the necessary
# formats for use in training the networks within the project.
#
# Note: YouTube-DL needs to be downloaded and installed. It can be found at:
#           https://github.com/rg3/youtube-dl
#
#       and installed with the following command on a linux machine:
#           git clone https://github.com/rg3/youtube-dl &&
#           cd youtube-dl &&
#           python2 setup.py install

from __future__ import unicode_literals
import youtube_dl

def main():
    # Get the save location and URL's for the youtube video's to download
    video_location = '../../data/videos/'
    #video_location = '../../data/short_videos/'
    #youtube_url_file_location = 'short_videos.txt'
    youtube_url_file_location = 'files.txt'
    ydl_opts = get_youtube_dl_options("long", video_location)

    download_videos(video_location, youtube_url_file_location, ydl_opts)

def get_youtube_dl_options(short_or_long, video_location):
    # Return the download options for YouTube-DL depending on quality requirements

    short_low_quality = {
            'restrictfilenames': True,
            'nopart': True,
            'ignoreerrors': True,
            'format': '133/134',
            'outtmpl': video_location + '%(autonumber)s.%(ext)s'
    }
    
    long_high_quality = {
            'restrictfilenames': True,
            'nopart': True,
            'ignoreerrors': True,
            'min_filesize': 400000000,
            'format': '136/best[height=720]',
            'outtmpl': video_location + '%(autonumber)s.%(ext)s'
    }

    if short_or_long == "short":
        return short_low_quality

    if short_or_long == "long":
        return long_high_quality


def download_videos(video_location, youtube_url_file_location, ydl_opts):
    youtube_urls = [line.rstrip('\n') for line in open(youtube_url_file_location)]


    # Download the files to the specified location
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_urls)

if __name__ == "__main__":
    main()
