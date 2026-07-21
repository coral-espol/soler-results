# Estructura de resultados

Organización de este repositorio alineada a las secciones experimentales de:

> "Learn alone or learn together? A minimal model of social learning in robot swarms" (2026)

## Convención de nombres

- `individual_learning` / `social_learning` — los dos modelos de aprendizaje del paper (Sección III.C).
- `greedy` / `selective` — las dos estrategias de aceptación de tareas (Sección III.D).
- `case1` / `case2` / `case3` — los tres casos de la validación del modelo (0, 1 y 3 peers observados; Sección III.C.1).

## Mapa de carpetas

```
validation/                        Validación del modelo (robot único, estático) — Sección III.C.1
  physical/                        5 rep. x 600 s, por caso
    case1/ case2/ case3/
      raw/                         datos crudos por corrida (qupa_XX.csv), un folder por run id
      processed/                   experiment_data.csv procesado por caso
    comparativa_casos/             comparativa de los 3 casos (estrategia 'both')
    comparativa_global/            comparativa de performance/f-measure/evolución de 'm'
    processing_data_selective/     comparativa de los 3 casos, solo estrategia selective
  scripts/                         scripts para regenerar las figuras de esta carpeta
  sim/                             (vacío — no se encontraron datos de validación en simulación)

experiment_physical/                Experimento 1: swarm de 8 robots físicos — Sección IV
                                     5 rep. x 1800 s, ventanas de 180 s
  individual_learning/
    greedy/runs/  selective/runs/   datos crudos por corrida
    experiment_data.csv             datos consolidados del modelo
    processing_data/                figuras/plots procesados
  social_learning/                  (misma subestructura)
  scripts/                          script para regenerar processing_data/ de cada modelo

experiment_simulation/              Experimento 2: mismas 4 condiciones en ARGoS — Sección V
                                     10 rep. x 10000 s, ventanas de 1000 s
  individual_learning/{greedy,selective}/
  social_learning/{greedy,selective}/
                                     (vacío — pendiente de correr/exportar las simulaciones)
```

## Cómo regenerar las gráficas

Requisitos (una sola vez): `pip install -r requirements.txt` desde la raíz de este repo.

**Validación** (`validation/scripts/`):
```
cd validation/scripts
python casos_data.py                  # comparativa_global: performance, f-measure, evolución de 'm'
python data_process_EXP.py both       # comparativa_casos: histogramas, tendencia, violin, f-measure, scatter
python data_process_EXP.py selective  # mismo set de gráficas, filtrando solo robots selective
```
(`greedy` no aplica: los datos de validación no tienen corridas con esa estrategia.)

**Experimento 1 físico** (`experiment_physical/scripts/`):
```
cd experiment_physical/scripts
python run_data_process.py                     # procesa individual_learning y social_learning
python run_data_process.py individual_learning # o solo uno de los dos modelos
python run_data_process.py social_learning
```

Cada script lee las rutas relativas a su propia ubicación (`../physical` o `../<modelo>`), así que
funcionan sin importar dónde clones este repositorio.

**Experimento 2 (simulación):** aún no hay script porque no hay datos de ARGoS. Cuando se agreguen los
logs de simulación en `experiment_simulation/<modelo>/<estrategia>/`, se puede adaptar
`run_data_process.py` de la misma forma (mismo formato de columnas, ajustando `ticks_per_sec`,
`window_sec` y `max_time_sec` a 1000 s / 10000 s según la Tabla III del paper).
