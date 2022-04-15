# NCA parameters
ITERATIONS = 50
GRID_SIZE = 25 

# NCA parameters - constant
N_CHANNELS = 2
N_HIDDEN_NODES = 0
N_HIDDEN_LAYERS = 0
NEIGHBORHOOD = 4 # Von Neumann (4) vs. Moore (8)
CONTINUOUS_SIGNALING = False 
MEMORY = False

# Flags
DIFFUSE = True
DIFFUSION_RATE = 0.5  # 50% of signal diffuses out of the cell each time step

# Evolution parameters
POP_SIZE = 10
GENERATIONS = 2
SAVE_ALL = False
objectives = ['error', 'error_phase1', 'error_phase2', 'MI'] 
# error = loss
# MI = empowerment

# Visualizations
CMAP_CELLS = 'binary'
CMAP_SIGNAL = 'gray'
CMAP_EMPOWERMENT = 'Blues'
TARGET_OUTLINE_COLOR = 'green'
CELL_STATE_OUTLINE_COLOR = 'm'
color_treatment_dict = {"random" : "tab:red",
                        "error_phase1_error_phase2" : "tab:orange",
                        "error" : "tab:blue",
                        "error_MI" : "tab:green",
                        "MI" : "tab:purple",
                        }