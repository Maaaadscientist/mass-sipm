import os
import sys
import ROOT

# Open your ROOT file
root_file_path = sys.argv[1]
file = ROOT.TFile(root_file_path, "READ")

def get_canvas_name():
    run = input("Enter run number: ")
    tile = input("Enter tile number: ")
    ch = input("Enter ch number: ")
    ov = input("Enter ov number: ")
    
    return f"charge_fit_run_{run}/charge_spectrum_tile_{tile}_ch_{ch}_ov_{ov}"

while True:
    canvas_name = get_canvas_name()
    
    # Retrieve the canvas from the ROOT file
    canvas = file.Get(canvas_name)
    
    if not canvas:
        print(f"Canvas with name {canvas_name} not found.")
        continue

    # Draw the canvas
    canvas.Draw()

    cont = input("Do you want to continue? (yes/no): ").lower()
    if cont == 'no':
        break

file.Close()

