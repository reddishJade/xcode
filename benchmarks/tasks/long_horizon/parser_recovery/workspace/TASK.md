# Configuration parser repair

`src/config_parser.py` parses a small `KEY=VALUE` format. Repair its failing
behavior without changing the public import surface.

Required behavior:

- trim surrounding whitespace from keys and values;
- ignore blank lines and lines whose first non-space character is `#`;
- split a setting at the first `=` so values may contain more `=` characters;
- raise `ParseError` for a non-comment line without `=` or with an empty key;
- preserve last-assignment-wins behavior for duplicate keys.

Constraints:

- modify `src/config_parser.py` as needed;
- do not modify `src/public_api.py`;
- do not modify `docs/compatibility.md`;
- do not create a second parser implementation such as `src/parser_v2.py`.

Success is determined only by the full unittest command in the task manifest.
