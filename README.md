UAS Casualty Triage Pathfinder

made this for the UAS-DTU round 2 task. rover has to find casualties on a map image, figure out priority order to visit them, and pathfind to each one using A*, then to the end point.

how to run: python final_uas.py --input uas --outdir output

(uas = folder with the map images, output = where results go, defaults to "output" if you dont pass it. can also just pass a single image path instead of a folder)

need opencv and numpy: pip install opencv-python numpy

what it outputs per image:

a mask png (black = obstacle, white = walkable)
a path png with the route drawn on it, casualties circled
a json with all the coords/scores/time
if you give it a folder with multiple images it also makes a ranking.json comparing all of them by score and by time.

some notes - the colour matching for terrain/markers uses squared distance from fixed BGR values, picked using paint's colour picker on the sample images. had to loosen the tolerance a lot because jpg compression messes with the colours more than expected, exact match was failing on stuff that looked fine to the eye. also added a morphological closing step on the mask to fill small gaps, otherwise A* would randomly fail to connect casualties that were basically right next to each other.

priority score = age score (shape) x severity score (colour). casualty score also factors in how far the casualty is from start vs how far the rover actually travelled to reach it.

if more than 8 casualties it stops brute forcing every order and just does a greedy sort + local swaps instead, faster but not guaranteed to be the actual best.

if start/end triangle isnt found the image just gets skipped, shows up in ranking.json under failed_images instead of crashing the whole batch.
