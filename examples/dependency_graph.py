from astscribe import NotebookAnalyzer


notebook = NotebookAnalyzer()

notebook.add_cell(
    """
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
"""
)

notebook.add_cell(
    """
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True)

tokenized = dataset.map(preprocess, batched=True)
"""
)

notebook.add_cell(
    """
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("example/model")
"""
)

notebook.add_cell(
    """
from transformers import Trainer

trainer = Trainer(model=model, train_dataset=tokenized)
trainer.train()
"""
)

print(notebook.render_dependency_graph())
print()
print(notebook.dependency_dot())

# Structured graph data is available without a visualization dependency.
graph = notebook.dependency_graph()
for edge in graph.edges:
    print(edge.producer_cell, "->", edge.consumer_cell, edge.symbols)
