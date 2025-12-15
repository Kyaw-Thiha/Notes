# Convert to Gif

`Install ffmpeg`

`Run the following command`
```bash
ffmpeg -i short-version.mp4 -vf "fps=15,scale=640:-1:flags=lanczos" -loop 0 output.gif
```

