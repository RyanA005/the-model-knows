# the model knows

llm based ctf challenge. participants get a small chat client and a
hugging face checkpoint that refuses to say the spooky vault flag in natural
chat. the flag is still recoverable from the weights.

participant info: [docs/participant.md](docs/participant.md)

organizer info: [docs/organizer.md](docs/organizer.md)

## run

put a merged hf tree under `model/` (see organizer docs), then:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python chat.py
```

checks:

```bash
python validate.py
python tools/solve.py
```

development flag is in `flag.txt`.
