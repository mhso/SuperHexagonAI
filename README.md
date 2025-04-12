# SuperHexagonAI
Self-playing program that can beat the hardest level of the video game Super Hexagon with no knowledge of the game's code using Computer Vision and a rule-based approach using simple heuristics to determine what moves to make.

[Click here for a showcase and a breakdown of how it works.](https://youtube.com)

![Banner Image](banner.png "Super Hexagon AI")

## Approach
The program works by taking screenshots of the game and extracting necessary information with the help of OpenCV and general image analysis tricks. This way, the program detects the cursor and all the obstacles we wish to avoid and splits the screen into six sectors where obstacles can appear.

Moves are made by assigning each sector a risk score based on the obstacles present and how long it would take to move to that sector. The sector with the lowest score is where we go next. Simple as that.

Some obstacles are detected explicitely and a hardcoded sequence of moves are used for those.

If obstacles form a shape where there is only exit, we instead go straight to that exit.

## Installation
If you wish to try the program out for yourself, simply clone the repository and install the requirements. Bear in mind, that it only works on Windows for now, because of reliance on `pywin32` for screenshotting (but that might change).

The program is run through the `main.py` file. Open Super Hexagon, navigate to the level screen for the 'Hexagonest' level, run `main.py`, press 'Space' when prompted, and the game will play itself. Running Super Hexagon in windowed mode is recommended, as the program struggles to keep up with every frame when running in fullscreen (depending on the hardware used).

I take no responsibility if you get in trouble for cheating the leaderboards, but the game is so old now I doubt it matters.