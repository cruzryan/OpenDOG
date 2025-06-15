import os
import sys
import numpy as np

from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

class RealTimePlotter:
	def __init__(self, buffer_size=500):
		self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
		self.win = pg.GraphicsLayoutWidget(title="Real-time Contact Forces")
		self.win.resize(1200, 800)
		self.plots = []
		self.curves_actual = []
		self.buffer_size = buffer_size
		self.data_curves = [np.zeros(self.buffer_size) for _ in range(4)]
		paw_names = ['Front Left', 'Front Right', 'Back Left', 'Back Right']
		for i, name in enumerate(paw_names):
			p = self.win.addPlot(title=name)
			p.setLabel('left', 'Force (Z)')
			p.setLabel('bottom', 'Samples')
			p.setYRange(0, 20)
			p.addLegend()
			actual_curve = p.plot(self.data_curves[i], pen=pg.mkPen('y', width=2), name='Actual')
			self.plots.append(p)
			self.curves_actual.append(actual_curve)
		self.win.show()
	
	def update_plot(self, new_samples):
		for i in range(4):
			self.data_curves[i] = np.roll(self.data_curves[i], -1)
			self.data_curves[i][-1] = new_samples[i]
			self.curves_actual[i].setData(self.data_curves[i])
		
		# Esta línea es crucial para que la gráfica se actualice durante el bucle
		self.app.processEvents()
	
	def close(self):
		"""Cierra la ventana de la gráfica y finaliza el programa de forma forzosa."""
		print("Cerrando la ventana de la gráfica...")
		self.win.close()
		# Usamos os._exit(0) para garantizar que el proceso muera y no se quede colgado
		print("Finalizando el programa.")
		os._exit(0)

