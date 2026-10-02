from astscribe import NotebookAnalyzer


notebook = NotebookAnalyzer.from_ipynb("training.ipynb")

print(notebook.render_methodology(include_evidence=True))

if notebook.skipped_cells:
    print("\nSkipped cells:")
    for skipped in notebook.skipped_cells:
        print(f"- cell {skipped.index}: {skipped.reason}")
