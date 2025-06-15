import os
import sys
import numpy as np
import time # Para el ejemplo

from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

class RealTimePlotter:
	def __init__(self, buffer_size=500):
		self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
		
		# Cambiamos el título a algo más general
		self.win = pg.GraphicsLayoutWidget(title="Telemetría del Robot en Tiempo Real")
		self.win.resize(8000, 600) # Hacemos la ventana un poco más grande
		
		self.buffer_size = buffer_size
		
		# --- ESTRUCTURA DE DATOS MODIFICADA ---
		# Usamos diccionarios para manejar múltiples grupos de gráficas
		self.plot_widgets = {}
		self.curve_groups = {}
		self.data_groups = {}

		# --- GRÁFICA 1: VELOCIDAD DEL ROBOT ---
		# La colocamos en la primera fila (row=0), ocupando 4 columnas (colspan=4)
		p_velocity = self.win.addPlot(title="Velocidad del Robot (eje X)", row=0, col=0, colspan=4)
		p_velocity.setLabel('left', 'Velocidad (m/s)')
		p_velocity.setLabel('bottom', 'Muestras')
		p_velocity.setYRange(-2, 2) # Rango de ejemplo para la velocidad
		p_velocity.addLegend()
		
		# Inicializamos los datos y la curva para la velocidad
		self.data_groups['velocity'] = [np.zeros(self.buffer_size)]
		# Asumimos una sola curva para la velocidad en X
		velocity_curve = p_velocity.plot(self.data_groups['velocity'][0], pen=pg.mkPen('g', width=2), name='Velocidad X')
		self.curve_groups['velocity'] = [velocity_curve]


		# --- GRÁFICAS 2: FUERZAS DE CONTACTO ---
		# Las colocamos en la segunda fila (row=1)
		self.data_groups['forces'] = [np.zeros(self.buffer_size) for _ in range(4)]
		self.curve_groups['forces'] = []
		
		paw_names = ['Front Left', 'Front Right', 'Back Left', 'Back Right']
		for i, name in enumerate(paw_names):
			# Creamos cada gráfica en la fila 1, en su columna correspondiente (col=i)
			p = self.win.addPlot(title=name, row=1, col=i)
			p.setLabel('left', 'Fuerza (Z)')
			p.setLabel('bottom', 'Muestras')
			p.setYRange(0, 20)
			p.addLegend()
			
			actual_curve = p.plot(self.data_groups['forces'][i], pen=pg.mkPen('y', width=2), name='Fuerza Z')
			self.curve_groups['forces'].append(actual_curve)
			
		self.win.show()
	
	# --- FUNCIÓN DE ACTUALIZACIÓN MODIFICADA ---
	def update_plots(self, force_samples=None, velocity_samples=None):
		"""
		Actualiza los grupos de gráficas con los nuevos datos proporcionados.
		
		Args:
			force_samples (list/array): Lista de 4 valores para las fuerzas de contacto.
			velocity_samples (list/array): Lista de 1 valor para la velocidad en X.
		"""
		
		# Actualizar las gráficas de fuerza si se proporcionaron datos
		if force_samples is not None:
			group_data = self.data_groups['forces']
			group_curves = self.curve_groups['forces']
			for i in range(len(group_curves)):
				group_data[i] = np.roll(group_data[i], -1)
				group_data[i][-1] = force_samples[i]
				group_curves[i].setData(group_data[i])

		# Actualizar la gráfica de velocidad si se proporcionaron datos
		if velocity_samples is not None:
			group_data = self.data_groups['velocity']
			group_curves = self.curve_groups['velocity']
			# Como solo hay una curva de velocidad, usamos el índice 0
			group_data[0] = np.roll(group_data[0], -1)
			group_data[0][-1] = velocity_samples[0]
			group_curves[0].setData(group_data[0])
			
		# Procesar los eventos de la GUI para que se redibuje todo
		self.app.processEvents()
	
	def close(self):
		"""Cierra la ventana de la gráfica y finaliza el programa de forma forzosa."""
		print("Cerrando la ventana de la gráfica...")
		self.win.close()
		print("Finalizando el programa.")
		os._exit(0)

