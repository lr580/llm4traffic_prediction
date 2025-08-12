import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDatasetHandler
handler = PEMSDatasetHandler(3, loadData=True)
handler.plotData(0, 288*35, 165, (16, 6))
handler.plotData(0, 288*35, 166, (16, 6))
handler.plotData(0, 288*35, 233, (16, 6))
handler.plotData(0, 288, 100)
handler.plotData(288*2, 288*3, 100)
handler.plotData(288*2, 288*3, 200)
handler.plotData(288*7, 288*14, 58, (14, 6))