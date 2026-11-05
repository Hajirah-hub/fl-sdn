# fedsdn

Implementation of Section III of `docs/paper.pdf`: aggregation strategies for federated threat
detection in multi-controller SDN under controller-level non-IID traffic.

Setup (Windows, Python 3.13):

    py -3.13 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e . --no-deps

Raw data goes in `data/raw/insdn/`. See `CLAUDE.md` for data formats, rules and run configs.
