# emd_data_generator_blender

A simple system for generating cloth manipulation data using Blender.
Coded by Solvi Arnold and used for data generation in the following publications:

[1] Solvi Arnold and Kimitoshi Yamazaki, "Fast and Flexible Multi-Step Cloth Manipulation Planning Using an Encode-Manipulate-Decode Network (EM*D Net)", Frontiers in Neurorobotics, vol. 13, 2019, doi:10.3389/fnbot.2019.00022.

[2] Daisuke Tanaka, Solvi Arnold and Kimitoshi Yamazaki, "EMD Net: An Encode–Manipulate–Decode Network for Cloth Manipulation", IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 1771-1778, July 2018, doi:10.1109/LRA.2018.2800122.

The code was revised to work with more recent Blender versions.
Tested with Blender version 2.91.2.


======================================
=== Purpose ==========================

The system generates single- and dual-handed cloth manipulation examples on a square cloth. Each example consists of a sequence of random manipulations, starting from a single initial state. Each manipulation is defined by one or two grasp points and a displacement vector. The manipulator object grasps the cloth at the grasp points, moves the grasp points along the displacement vector, and then drops them (see [1] for details). The system then waits for a set number of frames so the cloth shape can stabilise, and then a snapshot of the shape is taken. Manipulation sequences and the shape sequences they produced are stored to the dataset. The data can be used to e.g. train planning systems, as we do in the papers listed above.


======================================
=== Usage ============================

[A. Data generation]

1) Ensure you can see Blender's terminal output (e.g. by launching Blender from a terminal).
2) Open the scene "cloth_manipulation.blend". You should see a square turqoise cloth on a square grey work surface.
2) From Blender's text file editor, open manipulation_addon.py and run it (▶ button). This should produce a "Manipulation" tab on the main view.
3) In the Manipulation tab, set the "Base path" field to a work directory containing "init.pickle". Data will be generated to sub-directories of this work directory. The "init.pickle" file contains the initial cloth shape from which all manipulation sequences will start (by default this state has the cloth fully spread out with its axes aligned with the axes of the workspace).
4) In the "Data dir" field, enter a name for the directory to store generated data to. Data files will be stored to base_path/data_dir.
5) In the "Data generation" section of the Manipulation tab, find the following settings:
- Sequence length: The sequence length for a single example (i.e. the number of manipulations per manipulation sequence).
- Number of Files: The total number of data files to generate.
- Examples per file: The number of examples in each individual file (generally 1 is fine here).
- Name base: Generated files are named as name_base000000.npz. Can be useful to e.g. differentiate files when performing data generation in parallel with multiple Blender instances.
6) Press "Generate data" to start the data generation process.
7) Check the terminal output. Errors and problems are reported here, as well as the progress of the data generation process. Data generation can take a lot of time, and the Blender GUI will give no indication of data generation being started or in progress (manipulations being performed are not visualised in the GUI). The GUI will be unresponsive while data is being generated. In our experience Blender's cloth simulation is not 100% stable. Check the progress of the data generation process every once in a while and terminate/resume as necessary.


[B. Manipulation settings]

Some aspects of the manipulation motion can be set from the Manipulation tab.
Note that not all combinations of settings are functional, and we did not test alternative settings much.
- Periodic: Enables/disables periodic boundaries in the way manipulations are defined. This is specific to how our planning system handles space, so you probably want to leave this off.
- Rounded: When enabled, manipulation trajectories are round arc segments (recommended). When disabled, trajectories consist of straight line segments (first a diagonal motion segment lifting the grasp points to a fixed height, followed by a horizontal motion segment bringing the grasp points to their drop positions).
- Polar: Sets whether the displacement vector for random manipulations is generated using polar or Cartesian coordinates. When using polar coordinates, the angle is drawn from [0, 360] and the length from [0, max_movement], so we have a circular domain. When using Cartesian coordinates, we draw the x and y component independently from [-max_movement, max_movement], so we have a square domain.
- Edge mode: When edge mode is enabled, only points near the edge of the cloth are selected as grasp points.
- Edge width: Sets the width of the edge for the edge mode setting above.
- Pick top: When enabled, we only pick the top layer when multiple layers of cloth overlap at the (x, y) coordinates defining a grasp point. When disabled, all layers are grasped.
- Motion speed: Scales the speed of the manipulation motion
- Pick-up range: Sets how close vertices have to be to the grasp point coordinates in order to get pinned to the manipulator object. Larger settings can be used to emulate larger gripper fingers.
- Stabilisation frames: The number of frames the simulation is left to run after the cloth is released. This time serves to let the cloth shape stabilise.
- Maximum displacement: Maximum length for the displacement vector in randomly generated manipulations.


[C. Direct controls]

The "Direct controls" section of the Manipulation tab can be used to experiment with various settings and see how they play out.

- Initialise cloth manipulation: Reverts the scene (and the addon's internals) to the starting state.
- Run one random manipulation: Generates a random manipulation using the current settings and performs it.
- Run specified manipulation: performs a user-specified manipulation (see below).

Note that the Blender GUI is non-responsive while a manipulation is calculated (you can see the calculation's progress in the terminal). Manipulations can be viewed by pressing play on the timeline editor once calculation is finished. Note that the number of manipulations that can be performed in sequence is limited by the "Sequence length" setting in the "Data generation" section of the tab.

User-specified manipulation set-up:
- Left active: sets whether the left hand is used.
- Right active: sets whether the right hand is used.
- x, y: Coordinates of the grasp points (one pair for each hand). Note that the cloth spans from (-0.7, -0.7) to (0.7, 0.7) by default.
- displacement x, y: Defines the displacement vector. For dual-handed manipulations, both hands use the same displacement vector.

Additionally, the cloth can be rotated:
- Rotate cloth: applies a rotation around the z-axis.
- degrees: sets the angle to rotate by

The "Save current state as initial state" button (marked '!!') saves the current shape of the cloth as the new initial shape. This overwrites the init.pickle file and cannot be undone (not even by restarting Blender). Don't remember why it's there. Maybe just better don't press it.


[D. Misc. functionalities]

The "Play back stored sequences" section of the Manipulation tab allows playback of previously generated data and manipulations generated by our planning system. The "Ping-Pong mode" section exists for operating alongside our planning system (interleaved planning and execution, as performed in the papers). These functionalities are not intended for stand-alone use so we do not detail them here.


======================================
=== Data format ======================

Data files are stored in base_path/data_dir/base_nameXXXXXX.npz (base_path, data_dir, base_name are set from the Manipulation tab, see above), with XXXXXX being an index number.
These files are Numpy archives containing three arrays each: 'states', 'picks', and 'moves'.

Data can be read from Python as follows:
  
  import numpy as np
  data = np.load('file_name')
  states = data['states']
  picks = data['picks']
  moves = data['moves']

"States" is an array of shape [N,(L+1),6561,3] where N is the number of examples in this file and L is the sequence length (these are set from the addon, see above). The sequence of shapes for a given example runs from the shape before the first manipulation to the shape after the last manipulation, so for manipulation sequences of length L we obtain shape sequences of length L+1. 6561 is the number of vertices in the mesh (we use a grid-style uniform 81x81 mesh for the square cloth), and 3 corresponds to the vertex coordinates (x, y, z). So e.g. states[i,s] gives us the cloth shape obtained after the [s]th manipulation in the [i]th example in the file (a [6561,3] array of vertex coordinates).

"Picks" is an array of shape [N,L,2,2] containing the grasp point coordinates of the manipulation sequences. Picks[i,s,0] gives us the (x, y) coordinate of the grasp point for the "left" hand in the [s]th manipulation in the [i]th example (an array of shape [2]), and picks[i,s,1] gives the coordinates for the "right" hand (note however that there is no meaningful distinction between the hands). For single-handed manipulations, the grasp point coordinates for the unused hand will be NaN.

"Moves" is an array of shape [N,L,2] containing the displacement vectors. Indexing is similar to the picks array, except there is only one displacement vector for each manipulation.


======================================
=== Notes ============================

- We found it beneficial to run a number of instances in parallel to speed up data generation on multi-core machines. The "Name base" field can be used to differentiate data files from different instances.
- The system stores intermediate states as state[pidXXXXX].pickle (with XXXXX being a process identifier to avoid name collisions when running multiple instances in parallel). These files are not automatically removed, but can safely be deleted once operation is finished.
- This was the author's first attempt at writing a Blender addon. Don't take it as an example of good addon design.

