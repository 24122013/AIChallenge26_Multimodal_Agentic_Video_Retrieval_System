# Notebooks

The legacy public-submission launchers were removed because they encoded an old
TKIS/VKIS schema and non-canonical artifact paths. Current KIS/QA practice uses
the backend CLI and router contracts documented in the root `README.md`.

Any new notebook must write runtime artifacts only under `data/`, call canonical
backend services, preserve original `frame_index`, and must not claim TRAKE
support until that task is implemented.
