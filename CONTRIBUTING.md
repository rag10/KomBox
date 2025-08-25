# Contribuir a KomBox

## Configuración
1. Clona y crea entorno:
   ```bash
   pip install -e ".[dev,dae]"
   ```
2. Estilo y calidad:
   ```bash
   ruff check .        # lint
   black --check .     # formato
   mypy kombox         # tipado (opcional)
   ```

## Flujo
- Crea una rama: `git checkout -b feature/mi-feature`
- Acompaña tus cambios con tests.
- Asegúrate de que `pytest -q` pasa.
- Abre un PR describiendo el *por qué* y el *qué*.

## Tests
```bash
pytest -q
pytest -q tests/test_externals_and_errors.py::test_external_injection_ok
```
