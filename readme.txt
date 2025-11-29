Podcast Automation — readme.txt

Overview:
- Simple script to denoise, trim silence, normalize loudness, export processed audio, and optionally transcribe using Whisper.

Quick setup (PowerShell on Windows):

1) Activate the project's virtual environment (created as `.venv`).

PowerShell (recommended):
& ".\.venv\Scripts\Activate.ps1"

If you used CMD (not recommended here):
.venv\Scripts\activate.bat

2) Install Python dependencies:

pip install -r requirements.txt

3) Install FFmpeg (optional but recommended for MP3 export):

- If you have Chocolatey:
choco install ffmpeg -y

- Or download from https://ffmpeg.org/download.html and add the `bin` folder to your PATH.

4) Run the processor (single file):

python process_podcast.py -i input/my_recording.wav -o output --lufs -16

5) Run the processor on a folder:

python process_podcast.py -i input -o output --lufs -16

6) Transcription (optional):

python process_podcast.py -i input/my_recording.wav -o output --transcribe --whisper-model tiny

Notes and tips:
- The script creates `output/` with a processed WAV file; MP3 export will be attempted only if FFmpeg is available.
- `requirements.txt` contains the Python packages used; installing it inside the project's `.venv` is recommended.
- If `git push` to GitHub fails due to 2FA, create a Personal Access Token (PAT) and use it when prompted, or set up `git credential manager`.

Troubleshooting:
- "Couldn't find ffmpeg" warnings mean MP3 export may fail; install ffmpeg as above.
- If `whisper` fails, ensure `openai-whisper` is installed and check CPU/GPU compatibility.

Contact:
- Repo: https://github.com/AnmolBudhewar8995/podcast_automation

Example full session (copy/paste into PowerShell):

& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
choco install ffmpeg -y   # optional
python process_podcast.py -i input/my_recording.wav -o output --lufs -16
python process_podcast.py -i input/my_recording.wav -o output --transcribe --whisper-model tiny
