# Estructura de resultados

Organización de este repositorio alineada a las secciones experimentales de:

> "Learn alone or learn together? A minimal model of social learning in robot swarms" (2026)

## Convención de nombres

- `individual_learning` / `social_learning` — los dos modelos de aprendizaje del paper (Sección III.C).
- `greedy` / `selective` — las dos estrategias de aceptación de tareas (Sección III.D).
- `case1` / `case2` / `case3` — los tres casos de la validación del modelo (0, 1 y 3 peers observados; Sección III.C.1).

## Mapa de carpetas

```
01_validation/                     Validación del modelo (robot único, estático) — Sección III.C.1
  physical/                        5 rep. x 600 s, por caso
    case1/ case2/ case3/
      raw/                         datos crudos por corrida (qupa_XX.csv), un folder por run id
      processed/                  experiment_data.csv procesado por caso
    comparativa_casos/             comparativas entre case1/case2/case3
  sim/                             (vacío — no se encontraron datos de validación en simulación)

02_experiment1_physical/           Experimento 1: swarm de 8 robots físicos — Sección IV
                                   5 rep. x 1800 s, ventanas de 180 s
  individual_learning/
    greedy/runs/  selective/runs/  datos crudos por corrida
    experiment_data.csv            datos consolidados del modelo
    processing_data/               figuras/plots procesados
  social_learning/                 (misma subestructura)
  comparativa_global/              comparativa individual vs social learning
  processing_data_both/            procesado combinando greedy+selective
  processing_data_selective/       procesado solo de la estrategia selective

03_experiment2_simulation/         Experimento 2: mismas 4 condiciones en ARGoS — Sección V
                                   10 rep. x 10000 s, ventanas de 1000 s
  individual_learning/{greedy,selective}/
  social_learning/{greedy,selective}/
                                   (vacío — pendiente de correr/exportar las simulaciones)
```

## Origen de los datos

Los datos de `01_validation/physical` y `02_experiment1_physical` fueron copiados desde
`REAL/EXP-REALES-CASOS` (carpeta hermana, fuera de este repo), preservando el original.
Mapeo de nombres antiguos → nuevos:

| Original                          | Nuevo                                              |
|------------------------------------|-----------------------------------------------------|
| `ESTATICOS/case_1,2,3`            | `01_validation/physical/case{1,2,3}/raw`            |
| `ESTATICOS/CASO1,2,3`             | `01_validation/physical/case{1,2,3}/processed`      |
| `ESTATICOS/processing_data_both`  | `01_validation/physical/comparativa_casos`          |
| `INDIVIDUALES/GREEDY,SELECTIVE`  | `02_experiment1_physical/individual_learning/{greedy,selective}/runs` |
| `SOCIAL/GREEDY,SELECTIVE`        | `02_experiment1_physical/social_learning/{greedy,selective}/runs` |
| `COMPARATIVA_GLOBAL`              | `02_experiment1_physical/comparativa_global`        |
| `processing_data_both` (top)      | `02_experiment1_physical/processing_data_both`      |
| `processing_data_selective` (top) | `02_experiment1_physical/processing_data_selective`  |

`03_experiment2_simulation` quedó como esqueleto vacío porque no se encontraron datos de
simulación (ARGoS) en ninguna de las carpetas del proyecto al momento de crear esta estructura.
