# Real log:     make all              (after `make data`)
# Dev / CI:     make dev              (synthetic fixture, no network, no API key)
MODEL ?= claude-haiku-5-5
MODELS ?= $(MODEL)
N ?=

.PHONY: data load discover decisions eval replay report all dev test mcp

data:      ; uv run ap-lens download
load:      ; uv run ap-lens load
discover:  ; uv run ap-lens discover
decisions: ; uv run ap-lens decisions
eval:      ; uv run ap-lens eval --models $(MODELS) $(if $(N),--n $(N))
replay:    ; uv run ap-lens replay --model $(MODEL)
report:    ; uv run ap-lens report --model $(MODEL)
all: load discover decisions eval replay report

dev:
	APLENS_SOURCE=fixture uv run ap-lens fixture
	APLENS_SOURCE=fixture $(MAKE) load discover decisions
	APLENS_SOURCE=fixture uv run ap-lens eval --models ""
	APLENS_SOURCE=fixture uv run ap-lens report

test: ; uv run pytest -q
mcp:  ; uv run ap-lens-mcp
