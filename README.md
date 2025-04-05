# SuperHexagonAI
Self-playing program that can beat the hardest level of the video game Super Hexagon with no knowledge of the game's code using Computer Vision and a rule-based approach using simple heuristics to determine what moves to make.

![Banner Image](banner.png "Super Hexagon AI")

## Approach
The program works by taking screenshots of the game and extracting necessary information with the help of OpenCV and general image analysis tricks. This way, the program detects the cursor and all the obstacles we wish to avoid and splits the screen into six sectors where obstacles can appear.

Moves are made by assigning each sector a risk score based on the obstacles present and how long it would take to move to that sector. The sector with the lowest score is where we go next, simple as that.

If obstacles form a shape where there is only exit, we instead go to that.

## Installation
If you wish to try the program out for yourself, simply clone the repository and install the requirements. Bear in mind, that it only works on Windows for now.

The program is run through the `main.py` file. Open Super Hexagon, navigate to the level screen for the 'Hexagonest' level, run `main.py`, press 'Space' when prompted, and the game will play itself. I take no responsibility if you get in trouble for cheating the leaderboards, but the game is so old now, I doubt it matters.