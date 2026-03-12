import json
import pathlib

p = pathlib.Path(
    r"d:/dev/learning/vaaet/notebooks/02_production/traffic_analyzer.ipynb"
)
nb = json.loads(p.read_text(encoding="utf-8"))
changed = False

for cell in nb["cells"]:
    source = cell.get("source", [])
    src = "".join(source) if isinstance(source, list) else str(source)
    if src.startswith(
        "# Cell 2b — Annotated Video Output with Full HUD + Classification"
    ):
        src = src.replace(
            'print(f"   🧪 Rejected speeds this minute: {minute_quality["rejected"]} | recovered: {minute_quality["recovered"]}")',
            "print(f\"   🧪 Rejected speeds this minute: {minute_quality['rejected']} | recovered: {minute_quality['recovered']}\")",
        )
        src = src.replace(
            'print(f"   🧪 Filtered speeds: {minute_quality["rejected"]} | recovered tracks: {minute_quality["recovered"]}")',
            "print(f\"   🧪 Filtered speeds: {minute_quality['rejected']} | recovered tracks: {minute_quality['recovered']}\")",
        )
        cell["source"] = src.splitlines(keepends=True)
        changed = True
        break

if not changed:
    raise RuntimeError("Target cell not found")

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook quoting fixed")
