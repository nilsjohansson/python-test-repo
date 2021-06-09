from gluonPretrainedFasterRcnn import run as runRcnn
from gluonPretrainedSsdTutorial import run as runSsd
from gluonPretrainedYolo import run as runYolo
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../testdata/parkingoutside2-rotated.jpg"
abs_file_path = os.path.join(script_dir, rel_path)

# runRcnn(abs_file_path)
runSsd(abs_file_path)
# runYolo(abs_file_path)