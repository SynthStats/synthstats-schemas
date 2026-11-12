Output assets/demo.gif

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 20
Set Width 1200
Set Height 675
Set Padding 24
Set Framerate 30
Set WaitTimeout 180s
Set WaitPattern /^\$$/
Set Theme { "name": "SynthStatsDark", "background": "#0D0F17", "foreground": "#E9EBF5", "cursor": "#ABA6F2", "selection": "#242839", "black": "#151925", "red": "#D41950", "green": "#0096D0", "yellow": "#ABA6F2", "blue": "#0072B2", "magenta": "#5C56A5", "cyan": "#0096D0", "white": "#E9EBF5", "brightBlack": "#A0A6BC", "brightRed": "#F03268", "brightGreen": "#0096D0", "brightYellow": "#ABA6F2", "brightBlue": "#0096D0", "brightMagenta": "#ABA6F2", "brightCyan": "#0096D0", "brightWhite": "#FBFBFD" }

Hide
Type "export PS1='$ ' && clear"
Enter
Wait
Show

Type "uv venv --python 3.12"
Sleep 300ms
Enter
Wait+Screen /virtual environment/
Sleep 500ms
Type "uv pip install . 2>&1 | tail -3"
Sleep 300ms
Enter
Wait+Screen /zipp==/
Sleep 800ms
Type "clear"
Enter
Sleep 300ms

Type ".venv/bin/python - <<'PY'"
Sleep 300ms
Enter
Sleep 400ms
Type "from synthstats_schemas import canonical_json, parse_task_contract"
Enter
Sleep 120ms
Type ""
Enter
Sleep 120ms
Type "contract = parse_task_contract('''{"
Enter
Sleep 120ms
Type '  "schema_version": "synthstats.task_contract.v2",'
Enter
Sleep 120ms
Type '  "language": "pymc",'
Enter
Sleep 120ms
Type '  "entry_point": {"function": "model", "parameter": "data", "returns": "pymc.Model"},'
Enter
Sleep 120ms
Type '  "data": {'
Enter
Sleep 120ms
Type '    "container": "dict", "column_type": "numpy.ndarray",'
Enter
Sleep 120ms
Type '    "row_count_key": "n", "keys": ["x", "y"], "target_key": "y"'
Enter
Sleep 120ms
Type "  },"
Enter
Sleep 120ms
Type '  "provided_names": {"pm": "pymc", "np": "numpy"},'
Enter
Sleep 120ms
Type '  "conventions": [],'
Enter
Sleep 120ms
Type '  "environment": {'
Enter
Sleep 120ms
Type '    "image_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",'
Enter
Sleep 120ms
Type '    "python": "3.12", "packages": {}'
Enter
Sleep 120ms
Type "  },"
Enter
Sleep 120ms
Type '  "interaction": {"mode": "pool", "query_budget": 3, "observation_key": "y"},'
Enter
Sleep 120ms
Type '  "required_predictor_groups": [["x"]],'
Enter
Sleep 120ms
Type '  "relationship": "conditional_response",'
Enter
Sleep 120ms
Type '  "row_semantics": "matched_record"'
Enter
Sleep 120ms
Type "}''')"
Enter
Sleep 120ms
Type "encoded = canonical_json(contract)"
Enter
Sleep 120ms
Type "assert parse_task_contract(encoded) == contract"
Enter
Sleep 120ms
Type "print(contract.interaction_mode)  # pool"
Enter
Sleep 120ms
Type "PY"
Sleep 500ms
Enter
Sleep 3s
Wait+Screen /pool/
Sleep 3s
