# Contributing

```bash
pip install -e ".[dev]"
pytest -q               # unit tests and a real curses session in a pty
sampire --selftest      # the accuracy checks
```

Ground rules:

1. The audio callback (`Engine.render_block` and everything it calls) must not do I/O, take locks or wait.
2. Anything random takes a seed. Same inputs, same output.
3. New DSP needs a check in `sampire/selftest.py` that proves it does what it says.
4. Keep the core dependencies to NumPy, SciPy and SoundFile. Everything else is an optional extra.
5. No file reads or writes outside the project folder, `~/.config/sampire` and the paths the user typed.
