from gluonPretrainedFasterRcnn import run as runRcnn
from gluonPretrainedSsdTutorial import run as runSsd
from gluonPretrainedYolo import run as runYolo

fname = r"E:\\Temp\3parkedcars-noWindowInterference-rotated.jpg"
runRcnn(fname)
runSsd(fname)
runYolo(fname)