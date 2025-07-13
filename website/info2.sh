mkdir -p videos_640p

for f in videos/*.mp4; do
    base=$(basename "$f")
    ffmpeg -i "$f" -vf "scale=640:-2" -c:v libx264 -crf 23 -preset medium -profile:v high -level 4.0 -movflags +faststart -an "videos_640p/$base"
done
