mkdir -p videos_h264

for f in videos/*.mp4; do
    base=$(basename "$f")
    ffmpeg -i "$f" -c:v libx264 -crf 23 -preset fast -an "videos_h264/$base"
done
