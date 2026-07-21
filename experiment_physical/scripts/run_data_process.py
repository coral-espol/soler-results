"""
This script processes data collected from robot simulation experiments.
It contains functions to clean, transform, and prepare data for subsequent analysis.
SOLER PROJECT.
Author: Gabriel Madroñero
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import sys
from scipy import stats
import seaborn as sns
import math
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# === Global parameters (adjust if changed in the controller) ===
T_TICKS_PER_SEC = 10.0   # Should match T_TICKS_PER_SEC in Lua controller
W_STD_SEC = 180.0        # w_std from paper (for reference axis)
W_MIN_SEC = 24.0         # w_min from paper (for reference axis)
MIN_TIMESTEP = 0         # Adjusted to 0 to capture all new data
MAX_TIMESTEP = 1000000   # Maximum timestep to consider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def thousands_formatter(x, pos):
    """
    Formatter to display Y-axis values in thousands (x10^3).
    Example: 120000 -> 120
    """
    return f"{x/1000:.1f}"

class RobotDataProcessor:
    """Class to process robot simulation experiment data."""
    
    def __init__(self, csv_path: str, output_dir: str, ticks_per_sec: float = 10.0):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.ticks_per_sec = ticks_per_sec
        self.raw_data = None
        self.data_selective = None
        self.data_greedy = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            self.raw_data = pd.read_csv(self.csv_path)
            logger.info(f"Successfully loaded data with {len(self.raw_data)} rows")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        required_columns = ['tick', 'greedy', 'robot', 'm', 'p_x', 'planned_wticks', 'task', 'x', 'y', 'seed']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        if not pd.api.types.is_numeric_dtype(df['tick']):
            logger.warning("'tick' column should be numeric")
        if df.empty:
            logger.warning("DataFrame is empty")
            return False
        logger.info("Data validation passed")
        return True
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_clean = df.copy()

        # Mapeo de nomenclatura física (real) a lógica (BLUE/RED) que ya usan los
        # plots de este script. No existía en la versión de simulación porque el
        # controlador Lua ya logueaba 'BLUE'/'RED' directamente.
        if 'task' in df_clean.columns:
            df_clean['task'] = df_clean['task'].replace({'TYPE_A': 'BLUE', 'TYPE_B': 'RED'})

        logger.info(f"Filtering data: timestep {MIN_TIMESTEP} to {MAX_TIMESTEP}")
        df_clean = df_clean[(df_clean['tick'] >= MIN_TIMESTEP) & (df_clean['tick'] <= MAX_TIMESTEP)]

        if df_clean['greedy'].dtype == object:
            df_clean['greedy'] = df_clean['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
        
        data_selective = df_clean[df_clean['greedy'] == False].copy()
        data_greedy = df_clean[df_clean['greedy'] == True].copy()
        
        for data in [data_selective, data_greedy]:
            if not data.empty:
                data['time_seconds'] = data['tick'] / self.ticks_per_sec
                data['w_sec'] = data['planned_wticks'] / self.ticks_per_sec
        
        return data_selective, data_greedy
    
    def get_basic_stats(self, df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        if df.empty: return {}
        stats = {
            'strategy': strategy_name,
            'total_entries': len(df),
            'unique_robots': df['robot'].nunique(),
            'task_distribution': df['task'].value_counts().to_dict(),
            'avg_p_x': df['p_x'].mean(),
            'avg_m': df['m'].mean()
        }
        if 'w_sec' in df.columns:
            stats.update({'avg_completion_time': df['w_sec'].mean()})
        return stats
        
    def print_comparison(self):
        stats_selective = self.get_basic_stats(self.data_selective, "Selective")
        stats_greedy = self.get_basic_stats(self.data_greedy, "Greedy")
        print("\n" + "="*50)
        print("STRATEGY COMPARISON")
        print("="*50)
        for stats in [stats_selective, stats_greedy]:
            if stats:
                print(f"\n--- {stats['strategy']} Strategy ---")
                print(f"Total tasks: {stats['total_entries']}")
                print(f"Average p_x: {stats['avg_p_x']:.3f}")
                print(f"Average m: {stats['avg_m']:.3f}")
                print(f"Task Split: {stats['task_distribution']}")
                if 'avg_completion_time' in stats:
                    print(f"Avg completion time: {stats['avg_completion_time']:.2f} s")

    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================

    def plot_spatial_heatmap(self, save_path: Optional[str] = None) -> None:
        """Generates a 2D spatial heatmap of task execution locations."""
        if self.data_selective is None or self.data_greedy is None: 
            return
        
        logger.info("Generating Spatial Heatmap")
        
        # Figura de 24x12 para que cada uno de los 2 subplots tenga un área cercana a 12x12
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        bounds = [[-2.5, 2.5], [-2.5, 2.5]]

        for ax, data, title in zip([ax1, ax2], [self.data_selective, self.data_greedy], ["Selective Strategy", "Greedy Strategy"]):
            if data is not None and not data.empty:
                # Mapa de calor de fondo
                h = ax.hist2d(data['x'], data['y'], bins=25, range=bounds, cmap='Greys', alpha=0.3)
                
                blue_tasks = data[data['task'] == 'BLUE']
                red_tasks = data[data['task'] == 'RED']
                
                # Scatter plots: Aumentado el tamaño (s=60) para que los puntos destaquen en la figura grande
                ax.scatter(blue_tasks['x'], blue_tasks['y'], color='blue', s=60, alpha=0.7, 
                           label='BLUE Tasks', edgecolors='white', linewidth=0.8)
                ax.scatter(red_tasks['x'], red_tasks['y'], color='red', s=60, alpha=0.7, 
                           label='RED Tasks', edgecolors='white', linewidth=0.8)
                
                # Títulos y Etiquetas con fuentes grandes y negrita
                ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
                ax.set_xlabel('X Coordinate (m)', fontsize=20, fontweight='bold', labelpad=15)
                ax.set_ylabel('Y Coordinate (m)', fontsize=20, fontweight='bold', labelpad=15)
                
                # Limites y Aspecto
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                # CRÍTICO: Garantiza que 1 metro en X mida visualmente lo mismo que 1 metro en Y
                ax.set_aspect('equal', adjustable='box') 
                
                # Aumentar el tamaño de los números en los ejes
                ax.tick_params(axis='both', which='major', labelsize=16)
                
                # Grilla
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Leyenda bien definida
                ax.legend(loc='upper right', fontsize=16, framealpha=0.9, edgecolor='black')
                
                # Colorbar configurada para coincidir con la nueva escala de fuentes
                cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Activity Density', rotation=270, labelpad=25, fontsize=18, fontweight='bold')
                cbar.ax.tick_params(labelsize=14) # Tamaño de los números de la barra de color

        fig.suptitle("Spatial Distribution of Task Executions", fontsize=28, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_comparison_histograms(self, save_dir: Optional[str] = None) -> None:
        """Plot histograms for both strategies side by side for comparison."""
        if self.data_selective is None or self.data_greedy is None: 
            return
        
        # Figura de 24x12: al ser 2 columnas, cada subplot se sentirá como un cuadrado de 12x12
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        bins = np.arange(W_MIN_SEC, W_STD_SEC + 6, 6)
        
        max_freq = 0
        if not self.data_selective.empty:
            counts_sel, _ = np.histogram(self.data_selective["w_sec"].values, bins=bins)
            max_freq = max(max_freq, counts_sel.max())
        if not self.data_greedy.empty:
            counts_gre, _ = np.histogram(self.data_greedy["w_sec"].values, bins=bins)
            max_freq = max(max_freq, counts_gre.max())
            
        y_limit_normalized = max_freq * 1.15

        # Zip con colores ligeramente más atractivos (verde y naranja estándar de matplotlib)
        strategies_data = zip(
            [ax1, ax2], 
            [self.data_selective, self.data_greedy], 
            ["Selective Strategy", "Greedy Strategy"], 
            ['#2ca02c', '#ff7f0e'] 
        )

        for ax, data, title, color in strategies_data:
            if not data.empty:
                w_sec = data["w_sec"].values
                
                # Histograma
                ax.hist(w_sec, bins=bins, edgecolor="black", alpha=0.7, color=color)
                
                # Configuración de Ejes X e Y con fuentes grandes, negritas y espaciado (labelpad)
                ax.set_xlabel("Task completion time $w_x$ (s)", fontsize=20, fontweight='bold', labelpad=15)
                ax.set_ylabel("Number of tasks completed ($\\times 10^3$)", fontsize=20, fontweight='bold', labelpad=15)
                
                #ax.set_ylim(0, y_limit_normalized)
                ax.set_ylim(0, 20000)
                ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                
                # Aumentar tamaño de los números en ambos ejes
                ax.tick_params(axis='both', which='major', labelsize=16)
                
                # Títulos de cada Subplot
                ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
                
                # Líneas verticales: Añadido "linewidth" y "label" para que aparezcan en la leyenda
                ax.axvline(W_MIN_SEC, linestyle="--", color='red', linewidth=2.5, alpha=0.8, label='$W_{min}$')
                ax.axvline(W_STD_SEC, linestyle="--", color='blue', linewidth=2.5, alpha=0.8, label='$W_{std}$')
                
                # Grilla: Solo en Y para histogramas suele verse más limpio
                ax.grid(True, axis='y', linestyle='--', alpha=0.6)
                
                # Caja de texto: Aumento de tamaño, negrita y separador de miles para el conteo
                ax.text(0.03, 0.97, f"Total Tasks: {len(w_sec):,}", 
                        transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
                
                # Leyenda para explicar qué son las líneas roja y azul
                ax.legend(loc='center right', fontsize=16, framealpha=0.9, edgecolor='black')

        # Título principal aún más grande
        fig.suptitle("Comparison of Task Completion Times", fontsize=28, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        if save_dir: 
            plt.savefig(Path(save_dir) / "strategy_comparison_histograms.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_boxplot(self, window_sec: int = 1000, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        """Generates a boxplot showing tasks completed per time window for both strategies."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating improved performance boxplot")
        df = self.raw_data.copy()

        # 1. Processing and filtering data
        if df['greedy'].dtype == object:
             df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick'] / self.ticks_per_sec 

        # 2. Filtering and make tasks per time window
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]
        df['time_window'] = (np.ceil(df['time_sec'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        # 3. Data groping to count tasks per strategy and time window
        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()
        perf = df.groupby(['experiment_id', 'strategy', 'time_window']).size().reset_index(name='tasks_completed')

        # Origin to initialize boxplots with 0 tasks at time 0 for each experiment
        dummy_data = []
        for exp_id in perf['experiment_id'].unique():
            strat = perf[perf['experiment_id'] == exp_id]['strategy'].iloc[0]
            dummy_data.append({'experiment_id': exp_id, 'strategy': strat, 'time_window': 0, 'tasks_completed': 0})

        perf = pd.concat([perf, pd.DataFrame(dummy_data)], ignore_index=True)
        perf = perf.sort_values(by='time_window')

        # 4. Size of plot - Ajustado a 12x12 exacto según lo solicitado
        fig, ax = plt.subplots(figsize=(12, 12))

        perf['tasks_completed'] = perf['tasks_completed'].astype(int)
        # Color to diferrentiate strategies
        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'} # Azul y Naranja

        perf['time_window'] = perf['time_window'].astype(int)
        perf['time_window_str'] = perf['time_window'].astype(str)

        sns.boxplot(
            data=perf,
            x='time_window',
            y='tasks_completed',
            hue='strategy',
            hue_order=['selective', 'greedy'],  # orden fijo de la leyenda, igual que la referencia
            palette=my_palette,
            width=0.6,
            fliersize=6, # Ligeramente más grande para que se note en 12x12
            flierprops={'marker': 'o'},
            ax=ax
        )
        
        # Aumentar tamaño de los valores numéricos en los ejes X e Y
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Get current tick labels
        labels = [int(float(t.get_text())) for t in ax.get_xticklabels()]
        ticks = ax.get_xticks()

        ax.set_xticks(ticks)
        # Rotar ligeramente si son muchos valores (opcional) y asegurar tamaño de fuente
        ax.set_xticklabels([str(int(x)) for x in labels], fontsize=16, rotation=0)

        for i, box in enumerate(ax.patches):
            # 1. Get the color (RGBA) from actually box
            box_color = box.get_facecolor()

            # 2. Calculate the index of the corresponding flier line (outliers) for this box
            flier_line_idx = (i * 6) + 5 

            # 3. Apply the same color to the flier line (outliers) if it exists
            if flier_line_idx < len(ax.lines):
                flier_line = ax.lines[flier_line_idx]
                flier_line.set_markerfacecolor(box_color)  # Color strategy
                flier_line.set_markeredgecolor('black')    # Black border for visibility
                flier_line.set_alpha(0.7)                  # Opacity for visibility

        # 5. Customize axes, titles, and grid to match the paper's style
        # labelpad da un respiro visual entre los números del eje y el título del eje
        ax.set_xlabel("Time (s)", fontsize=24, fontweight='bold', labelpad=15)
        ax.set_ylabel("Number of tasks completed", fontsize=24, fontweight='bold', labelpad=15)
        
        # Opcional: Agregar un título general al gráfico si tu formato lo permite
        ax.set_title("Task completion over time", fontsize=24, fontweight='bold', pad=20)
        ax.set_ylim(0,50)
        # Hacer la grilla un poco más notoria pero elegante
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Mejorar la leyenda: tamaño de título, fondo blanco opaco para que no se cruce con las líneas
        legend = ax.legend(loc='upper left', fontsize=16, frameon=True, edgecolor='black')
        legend.set_title("Strategy", prop={'size': 18, 'weight': 'bold'})
        legend.get_frame().set_alpha(0.9)

        # save figure
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_search_time_distribution(self, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        """Generates a violin plot to show the overall distribution of task search times per strategy, including statistical legends."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating search time distribution plot with statistics")
        df = self.raw_data.copy()

        # 1. Initial data preparation
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick'] / self.ticks_per_sec
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]

        # Convert search ticks to seconds for better interpretability
        df['search_time_sec'] = df['search_ticks'] / self.ticks_per_sec

        # Calculate statistics for the legend ---
        stats = df.groupby('strategy')['search_time_sec'].agg(['mean', 'median', 'std', 'max']).round(2)

        # 2. VISUALIZATION
        # Cuadrado perfecto 12x12 para mantener consistencia con tus otros gráficos
        fig, ax = plt.subplots(figsize=(12, 12)) 

        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'}

        sns.violinplot(
            data=df,
            x='strategy',
            y='search_time_sec',
            order=['selective', 'greedy'], 
            palette=my_palette,
            inner='quartile', 
            cut=0,            
            linewidth=2.5,  # Aumentado de 1.2 a 2.5 para que los bordes y cuartiles destaquen en 12x12
            width=0.7,      # Ajusta el ancho para que los violines no se vean excesivamente gordos
            ax=ax,
            hue='strategy', 
            legend=False
        )

        # 3. AESTHETICS & LEGEND
        # Textos grandes, negritas y espacios (pad/labelpad)
        ax.set_title("Spent Task Search Time per Strategy", fontsize=24, fontweight='bold', pad=20)
        ax.set_ylabel("Search Time (s)", fontsize=20, fontweight='bold', labelpad=15)
        ax.set_xlabel("Strategy", fontsize=20, fontweight='bold', labelpad=15)

        # Aumentar tamaño de números en Y
        ax.tick_params(axis='y', labelsize=16)
        
        # Mejorar las etiquetas del eje X: Capitalizadas, grandes y en negrita
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Selective', 'Greedy'], fontsize=18, fontweight='bold')

        # Create custom legend handles with statistics ---
        legend_handles = []
        for strat in ['selective', 'greedy']:
            if strat in stats.index:
                s_mean = stats.loc[strat, 'mean']
                s_med = stats.loc[strat, 'median']
                s_std = stats.loc[strat, 'std']
                s_max = stats.loc[strat, 'max']

                # Format the text that will appear next to each color box
                label_text = f"{strat.capitalize()}\nMean: {s_mean}s\nMedian: {s_med}s\nStd: {s_std}s\nMax: {s_max}s"

                # Hacemos que el recuadro de color en la leyenda tenga un borde más definido (linewidth=1.5)
                patch = Patch(facecolor=my_palette[strat], edgecolor='black', linewidth=1.5, label=label_text)
                legend_handles.append(patch)
    
        # Place the legend in a spot where it doesn't obstruct the violins
        # CRÍTICO: labelspacing=1.3 evita que las líneas de texto dentro de un mismo bloque se peguen
        legend = ax.legend(handles=legend_handles, title="Descriptive Statistics", 
                           loc='upper right', fontsize=14, labelspacing=1.3,
                           framealpha=0.9, edgecolor='black')
        
        # Destacar el título de la leyenda
        legend.get_title().set_fontsize(16)
        legend.get_title().set_fontweight('bold')

        ax.set_ylim(0,1000)

        # Grilla horizontal más limpia para leer mejor los tiempos
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_f_measure_boxplot(self, window_sec: int = 1000, max_time_sec: int = 10000, save_path: Optional[str] = None) -> None:
        """Generates a boxplot of the F-measure (consistency of task execution) over time."""
        if self.raw_data is None or self.raw_data.empty:
            logger.warning("No data available to plot.")
            return

        logger.info("Generating improved F-measure boxplot")
        df = self.raw_data.copy()

        # 1. Initial data cleaning and preparation
        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})

        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})
        df['time_sec'] = df['tick'] / self.ticks_per_sec
        df = df[(df['time_sec'] >= 0) & (df['time_sec'] <= max_time_sec)]

        # Filter for valid tasks
        df = df[df['task'].isin(['BLUE', 'RED'])]

        # Define time windows
        df['time_window'] = (np.ceil(df['time_sec'] / window_sec) * window_sec).astype(int)
        df = df[(df['time_window'] > 0) & (df['time_window'] <= max_time_sec)]

        # Robust experiment detection using the 'seed' column
        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()

        # 2. VECTORIZED PROCESSING (Performance optimization)
        df = df.sort_values(by=['experiment_id', 'time_window', 'robot', 'tick'])
        df['prev_task'] = df.groupby(['experiment_id', 'time_window', 'robot'])['task'].shift(1)
        df['is_switch'] = (df['task'] != df['prev_task']) & (df['prev_task'].notnull())

        robot_stats = df.groupby(['experiment_id', 'strategy', 'time_window', 'robot']).agg(
            N=('task', 'count'),
            switches=('is_switch', 'sum')
        ).reset_index()

        robot_stats['f_measure'] = np.where(
            robot_stats['N'] == 1, 
            1.0, 
            1.0 - (2.0 * robot_stats['switches'] / robot_stats['N'])
        )

        f_df = robot_stats.groupby(['experiment_id', 'strategy', 'time_window'])['f_measure'].mean().reset_index()

        # 3. VISUALIZATION
        # Cuadrado perfecto 12x12
        fig, ax = plt.subplots(figsize=(12, 12))
        my_palette = {'selective': '#1f77b4', 'greedy': '#ff7f0e'}
        
        time_order = [0] + sorted(f_df['time_window'].unique().tolist())
        f_df['time_window'] = f_df['time_window'].astype(int)
        f_df['time_window_str'] = f_df['time_window'].astype(str)
        
        sns.boxplot(
            data=f_df,
            x='time_window',
            y='f_measure',
            hue='strategy',
            hue_order=['selective', 'greedy'],
            order=time_order,
            palette=my_palette,
            width=0.6,
            fliersize=6, # Aumentado para que los outliers destaquen en 12x12
            flierprops={'marker': 'o'},
            ax=ax
        )

        # Aumentar tamaño de números en los ejes antes de modificar los labels de X
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Format x-axis labels to scientific style [x10^3]
        labels = [int(float(t.get_text())) for t in ax.get_xticklabels()]
        new_labels = [f"{int(x/1000)}" if x != 0 else "0" for x in labels]
        ticks = ax.get_xticks()

        ax.set_xticks(ticks)
        # Aplicamos las nuevas etiquetas manteniendo el tamaño de fuente 16
        ax.set_xticklabels(new_labels, fontsize=16)

        # Custom logic to color the outliers based on their respective box color
        for i, box in enumerate(ax.patches):
            box_color = box.get_facecolor()
            flier_line_idx = (i * 6) + 5 
            if flier_line_idx < len(ax.lines):
                flier_line = ax.lines[flier_line_idx]
                flier_line.set_markerfacecolor(box_color)
                flier_line.set_markeredgecolor('black')
                flier_line.set_alpha(0.7)

        # 4. AESTHETICS
        # Añadido un título principal para dar contexto al gráfico
        ax.set_title("Specialization", fontsize=24, fontweight='bold', pad=20)
        
        # Ejes con fuentes grandes, negritas y espaciado
        ax.set_ylabel("F-measure ", fontsize=20, fontweight='bold', labelpad=15)
        ax.set_xlabel("Time (s) [×10³]", fontsize=20, fontweight='bold', labelpad=15)

        # Set y-limits: Aumenté el límite superior a 1.2 para que la leyenda grande no tape las cajas de valor 1.0
        ax.set_ylim(-0.1, 1.2) 
        
        # Grilla horizontal más limpia
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Leyenda robusta y opaca para que las líneas de la grilla no la crucen
        legend = ax.legend(loc='upper left', fontsize=16, framealpha=0.9, edgecolor='black')
        legend.set_title("Strategy", prop={'size': 18, 'weight': 'bold'})

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# SPECIALIZATION SCATTER PLOT 
# ============================================================================
class SpecializationScatterPlotter:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.raw_data = None
        self.robot_stats = None
        
    def load_data(self) -> pd.DataFrame:
        self.raw_data = pd.read_csv(self.csv_path)
        return self.raw_data

    def preprocess_data(self) -> pd.DataFrame:
        df = self.raw_data.copy()

        # Mapeo de nomenclatura física (real) a lógica (BLUE/RED), ver mismo
        # comentario en RobotDataProcessor.preprocess_data.
        if 'task' in df.columns:
            df['task'] = df['task'].replace({'TYPE_A': 'BLUE', 'TYPE_B': 'RED'})

        if df['greedy'].dtype == object:
            df['greedy'] = df['greedy'].astype(str).str.lower().map({'true': True, 'false': False})
        df['strategy'] = df['greedy'].map({True: 'greedy', False: 'selective'})

        df['experiment_id'] = df.groupby(['strategy', 'seed']).ngroup()

        TASK_TYPES = ['BLUE', 'RED']
        df_tasks = df[df['task'].isin(TASK_TYPES)].copy()
        
        # AJUSTE CLAVE: Añadimos 'seed' al groupby para no perder la información de la semilla
        self.robot_stats = (df_tasks.groupby(['experiment_id', 'strategy', 'seed', 'robot', 'task'])
                            .size().unstack(fill_value=0).reset_index())

        for task in TASK_TYPES:
            if task not in self.robot_stats.columns:
                self.robot_stats[task] = 0
                
        return self.robot_stats

    def _calculate_spec_index(self, df_subset: pd.DataFrame) -> float:
        if df_subset.empty: return 0.0
        blue = df_subset['BLUE']
        red = df_subset['RED']
        total = blue + red
        active = total > 0
        if not active.any(): return 0.0
        diff = (blue[active] - red[active]).abs()
        return (diff / total[active]).mean()

    # --- SCATTER PLOT---
    def plot_figure(self, save_path: str) -> None:
        if self.robot_stats is None: 
            return
        
        selective_data = self.robot_stats[self.robot_stats['strategy'] == 'selective']
        greedy_data = self.robot_stats[self.robot_stats['strategy'] == 'greedy']
        
        # Figura en cuadrado perfecto 12x12
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True, sharey=True)
        fig.patch.set_facecolor('#2b2b2b')

        def _plot(ax, data, title, show_xlabel=False):
            if data.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='white', fontsize=18)
                return
            
            total = data['BLUE'] + data['RED']
            bias = (data['BLUE'] - data['RED']) / total.replace(0, 1)
            
            ax.set_facecolor('#2b2b2b')

            # Ticks más grandes y legibles en color blanco
            ax.tick_params(colors='white', labelsize=14)
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
            # Scatter: Se aumentó el tamaño del punto (s=100) y se ajustó el 'norm' de -1 a 1
            scatter = ax.scatter(data['BLUE'], data['RED'], 
                                 c=bias, cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1),
                                 s=100, alpha=0.8, edgecolors='none')
            
            # Títulos y Ejes: Jerarquía de tamaños y negritas para destacar en 12x12
            ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
            ax.set_ylabel("Total tasks $\\tau_r$ (Red)", fontsize=18, fontweight='bold', labelpad=15)
            if show_xlabel:
                ax.set_xlabel("Total tasks $\\tau_b$ (Blue)", fontsize=18, fontweight='bold', labelpad=15)
            
            max_val = max(data['BLUE'].max(), data['RED'].max(), 10) * 1.1
            ax.plot([0, max_val], [0, max_val], color='white', linestyle='--', alpha=0.6, linewidth=2, label='Equilibrium')
            
            # CRÍTICO: Asegura que 1 unidad en X mida lo mismo que 1 en Y para que la diagonal sea a 45°
            ax.set_aspect('equal', adjustable='box')
            
            ax.grid(True, linestyle=':', alpha=0.3, color='white')
            
            # Agregamos la leyenda de la línea de equilibrio con estilo dark mode
            ax.legend(loc='upper left', fontsize=14, facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            
            # Caja de texto del índice de especialización: Fuente aumentada
            spec_idx = self._calculate_spec_index(data)
            ax.text(0.95, 0.95, f"Swarm Spec. Index: {spec_idx:.2f}", 
                    transform=ax.transAxes, ha='right', va='top', fontsize=16, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", alpha=0.9, edgecolor='none'))

        _plot(axes[0], selective_data, "Selective Strategy (SOLE-R)")
        _plot(axes[1], greedy_data, "Greedy Strategy (Baseline)", show_xlabel=True)

        # Ajuste de Colorbar para el tamaño 12x12
        # Aumenté un poco el ancho vertical de la barra (0.02 -> 0.025) para que no se vea tan delgada
        cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.025])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1)), 
                            cax=cbar_ax, orientation='horizontal')
        
        # Etiqueta de colorbar más grande y separada
        cbar.set_label('Robot Specialization: Red Specialist  <---  Generalist  --->  Blue Specialist',
                       color='white', fontsize=16, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(colors='white', labelsize=14)
        
        # Más espacio en 'bottom' para acomodar la colorbar que ahora tiene letras más grandes
        plt.subplots_adjust(bottom=0.18, hspace=0.25)
        
        if save_path:
            # facecolor es obligatorio aquí, de lo contrario la imagen exportada perderá el fondo oscuro
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        
        plt.show()

    # --- FUNCIÓN PARA LA MATRIZ DE EXPERIMENTOS ---
    def plot_strategy_grid(self, strategy: str, save_path: str) -> dict:
        """
        Plots a grid (e.g., 4x5) of scatter plots for each individual seed of a given strategy.
        Returns a dictionary with the best seed and its specialization index.
        """
        if self.robot_stats is None:
            print("Please run preprocess_data() first.")
            return {}

        df_strat = self.robot_stats[self.robot_stats['strategy'] == strategy]
        unique_seeds = df_strat['seed'].unique()
        n_experiments = len(unique_seeds)

        if n_experiments == 0:
            print(f"No data found for strategy: {strategy}")
            return {}

        # NORMALIZACIÓN: global_max garantiza ejes idénticos. Aumenté el margen a 1.1 para que no corten puntos
        global_max = max(self.robot_stats['BLUE'].max(), self.robot_stats['RED'].max(), 10) * 1.1

        # Cuadrícula dinámica
        cols = 5
        rows = math.ceil(n_experiments / cols)
        
        # Tamaño proporcional escalado (4 unidades por cada subplot para que no se vean apretados)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
        fig.patch.set_facecolor('#2b2b2b')  # Dark background
        
        axes = np.atleast_1d(axes).flatten()

        best_seed = None
        best_spec_idx = -1.0

        for i, seed in enumerate(unique_seeds):
            ax = axes[i]
            data = df_strat[df_strat['seed'] == seed]
            
            spec_idx = self._calculate_spec_index(data)
            
            if spec_idx > best_spec_idx:
                best_spec_idx = spec_idx
                best_seed = seed

            total = data['BLUE'] + data['RED']
            bias = (data['BLUE'] - data['RED']) / total.replace(0, 1)

            ax.set_facecolor('#2b2b2b')

            # CORRECCIÓN: cmap unificado a 'RdBu' y norm a vmin=-1, vmax=1.
            # Aumenté el tamaño del punto a s=50 para que resalte en el grid.
            ax.scatter(data['BLUE'], data['RED'], 
                       c=bias, cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1),
                       s=50, alpha=0.8, edgecolors='none')
            
            # Títulos y métricas por subplot (Fuentes escaladas)
            ax.set_title(f"Seed: {seed}", fontsize=16, fontweight='bold', color='white', pad=10)
            
            # Texto Spec Index: color negro para que se lea sobre el recuadro blanco
            ax.text(0.05, 0.95, f"Spec: {spec_idx:.2f}", 
                    transform=ax.transAxes, ha='left', va='top', fontsize=14, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", alpha=0.9, edgecolor='none'))
            
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white', labelsize=14)
            
            # Línea de equilibrio
            ax.plot([0, global_max], [0, global_max], color='white', linestyle='--', alpha=0.4, linewidth=1.5)
            
            ax.set_xlim(-1, global_max)
            ax.set_ylim(-1, global_max)
            
            # CRÍTICO: Garantiza que el grid sea perfectamente cuadrado (45 grados en la diagonal)
            ax.set_aspect('equal', adjustable='box')
            
            ax.grid(True, linestyle=':', alpha=0.3, color='white')

            # Etiquetas de ejes solo en los bordes
            if i % cols == 0:
                ax.set_ylabel("Red Tasks", fontsize=16, fontweight='bold', labelpad=10)
            if i >= (rows - 1) * cols:
                ax.set_xlabel("Blue Tasks", fontsize=16, fontweight='bold', labelpad=10)

        # Ocultar subplots vacíos
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Título principal de la figura
        fig.suptitle(f"Individual Experiments - {strategy.capitalize()} Strategy", 
                     fontsize=24, fontweight='bold', y=1.02, color='white') 

        # Barra de color global
        # Ajusté la posición para que no se empalme con el texto inferior
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1)), 
                            cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Robot Specialization: Red Specialist  <---  Generalist  --->  Blue Specialist',
                       color='white', fontsize=18, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(colors='white', labelsize=14)
        
        # Ajuste de márgenes
        plt.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.15)
        
        # Exportación forzando el modo oscuro
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            
        plt.show()

        print(f"[{strategy.capitalize()}] Best performing seed: {best_seed} with Spec Index: {best_spec_idx:.3f}")
        return {'strategy': strategy, 'best_seed': best_seed, 'best_spec_index': best_spec_idx}
        
# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def process_model(model_dir: Path):
    csv_path = model_dir / "experiment_data.csv"
    output_dir = model_dir / "processing_data"

    if not csv_path.exists():
        logger.warning(f"Skipping {model_dir.name}: no experiment_data.csv found at {csv_path}")
        return

    print("\n" + "#" * 60)
    print(f"# PROCESSING: {model_dir.name}")
    print("#" * 60)

    try:
        # ANTES: RobotDataProcessor(csv_path, output_dir) usaba el default
        # ticks_per_sec=10.0 (conversión de simulación). Los logs reales ya están
        # en segundos, así que se pasa ticks_per_sec=1.0 para no dividir dos veces.
        processor = RobotDataProcessor(csv_path, output_dir, ticks_per_sec=1.0)
        raw_data = processor.load_data()
        
        if not processor.validate_data(raw_data):
            return
            
        data_selective, data_greedy = processor.preprocess_data(raw_data)
        processor.data_selective = data_selective
        processor.data_greedy = data_greedy
        
        processor.print_comparison()
        
        print("\n" + "="*50)
        print("GENERATING PLOTS & FIGURES")
        print("="*50)

        # 1. Plot Comparison Histograms
        processor.plot_comparison_histograms(save_dir=output_dir)
            
        # 2. Plot Spatial Heatmap 
        processor.plot_spatial_heatmap(save_path=f"{output_dir}/spatial_heatmap.png")
            
        # 3. Figure 8 - Specialization Scatter Plot
        spec_plotter = SpecializationScatterPlotter(csv_path)
        spec_plotter.load_data()
        spec_plotter.preprocess_data()
        spec_plotter.plot_figure(save_path=f"{output_dir}/specialization_scatter.png")

        # 3.1 Gráficas matriciales y obtención de mejores semillas por experiento
        best_selective = spec_plotter.plot_strategy_grid('selective', save_path=f"{output_dir}/figure_grid_selective.png")
        best_greedy = spec_plotter.plot_strategy_grid('greedy', save_path=f"{output_dir}/figure_grid_greedy.png")

        print("--- Best experiments seeds ---")
        print("BEST SELECTIVE:", best_selective)
        print("BEST GREEDY:", best_greedy)

        # 4. Figure 6 - Performance Boxplot
        # ANTES: window_sec=1000, max_time_sec=10000 (escala de simulación).
        # INDIVIDUALES/SOCIAL duran 1800s reales.
        processor.plot_performance_boxplot(
            window_sec=180,
            max_time_sec=1800,
            save_path=f"{output_dir}/figure6_performance_boxplot.png"
        )

        # 5. Time search expend for the task distribution - Violin plot
        # ANTES: max_time_sec=10000 (escala de simulación).
        processor.plot_search_time_distribution(
            max_time_sec=1800,
            save_path=f"{output_dir}/figure_search_time_.png"
        )

        # 6. Figure 10 - F-measure Boxplot
        # ANTES: window_sec=1000, max_time_sec=10000 (escala de simulación).
        processor.plot_f_measure_boxplot(
            window_sec=180,
            max_time_sec=1800,
            save_path=f"{output_dir}/figure_f_measure_boxplot.png"
        )

    except Exception as e:
        logger.error(f"Error processing {model_dir.name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    # experiment_physical/ is the parent of scripts/, where this file lives.
    experiment_dir = Path(__file__).resolve().parent.parent

    # Pass a model name (individual_learning / social_learning) to process only
    # that one, e.g.: python run_data_process.py individual_learning
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = ["individual_learning", "social_learning"]

    for model in models:
        process_model(experiment_dir / model)


if __name__ == "__main__":
    main()