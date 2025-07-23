# zebrafish-tracker
## Installation
1. Install Python 3.10 - https://www.python.org/downloads/

2. Download Spinnaker SDK for Python 3.10, files available here: https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK - I've used versions 3.2 & 4.0 with success.

3. Download:
```
git clone git@github.com:camden-cummings/zebrafish-tracker.git
git submodule update --init
pip install requirements.txt
```

4. Test installation:
```
python gui_tracker.py
```
