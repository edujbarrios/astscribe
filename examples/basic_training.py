from astscribe import explain

code = """
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
"""

print(explain(code, style="scientific"))
