# Feynman Diagram Drawing Application

This application allows you to draw Feynman diagrams with various features. You can create and manipulate vertices, prongs, connections, and labels using a set of keyboard commands and mouse interactions.


## Setup

Make sure you're working in a python 3 environment with numpy and matplotlib installed. 
Also, when running pyplot, it should plot to the interactive window.  Then clone the project and run 
file feynman_diagrams.py. For instance,
```bash
git clone git@github.com:maadcoen/feynman_diagrams.git
python3 feynman_diagrams.py
```

## Usage

- Press any number (1 to 10) to create a vertex with the specified number of prongs.
- Select a vertex by clicking on its center.
- Select a prong by clicking on its end.
- Connect two prongs by clicking both ends sequentially.
- Select a connection by clicking on its middle.
- Move a vertex, prong, or bend a connection by clicking, holding, and moving.
- Change the appearance of a prong or connection by selecting it and pressing:
    - 'p': Photon line (wave)
    - 'h': Scalar boson (dashed)
    - 'g': Gluon line (spring)
    - 'i': Inwards arrow (spring)
    - 'o': Outwards arrow
    - '-': Flat

## Additional Commands

- 's': Add an initial state particle (one-prong to the right with the center indicated as a dot).
- 'f': Add a final state particle (one-prong to the left with the center indicated as a dot).
- 'q': Add a QED vertex.
- 'Q': Add a QCD vertex.
- 'e': Deselect all selected objects.
- 'd': Remove the selected connection or vertex.
- 'x': Remove everything in the diagram.

## Label Adding Mode

1. Select a vertex, a connection, or a prong.
2. Move the pointer to the desired location.
3. Press 't'. A cursor '/' will appear.
4. Start typing at the cursor (supports LaTeX commands if supported by matplotlib).
5. Press:
    - 'backspace': Remove the last added character.
    - 'enter': Leave label adding mode and keep the label.
    - 'escape': Leave label adding mode and discard the label.
    - 'up' or 'down': Enlarge or decrease fontsize
    - 'left' or 'right': Move cursor 
    - 'control': store text object into memory for copying
6. Select another vertex, connection, or prong and press 'control' to copy previously stored label

Labels can be clicked and moved like prongs and vertices and will move together with the object they label.

